"""Resize raw photographs and save an inspectable 21-image dataset per person."""
from pathlib import Path
import re

import pandas as pd
from PIL import Image

from .embeddings import read_rgb
from .pipeline import IMAGE_EXTENSIONS, UNKNOWN_FOLDERS, image_variants

EXPORT_VARIANTS = ('original', 'rotation_plus_8', 'horizontal_flip', 'grayscale',
                   'clahe', 'gamma_080', 'median_blur_3')


def resize_photos(source_dir, output_dir, max_side=1024, quality=95):
    """Write a copy of every photo whose longest side is at most ``max_side`` pixels.

    Phone rotation (EXIF) is applied first, the aspect ratio is kept, and photos
    that are already small enough are copied unchanged in size. The raw folder is
    never modified. Output: ``output_dir/<user_id>/<original stem>.jpg``. Unexpected
    files already in ``output_dir`` raise an error instead of being deleted.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    rows, planned = [], set()
    for path in sorted(source_dir.rglob('*')):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        relative = path.relative_to(source_dir)
        if len(relative.parts) != 2:
            raise ValueError(f'Use images/<user_id>/<filename>: {path}')
        target = output_dir / relative.parts[0] / (path.stem + '.jpg')
        if target in planned:
            raise ValueError(f'Two source files map to {target}; rename one of them')
        planned.add(target)
        image = Image.fromarray(read_rgb(path))
        original_size = image.size
        if max(original_size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format='JPEG', quality=quality)
        rows.append(dict(user_id=relative.parts[0].lower(), raw_path=str(path), resized_path=str(target),
                         original_width=original_size[0], original_height=original_size[1],
                         width=image.size[0], height=image.size[1],
                         resized=max(original_size) > max_side))
    extra = sorted(str(p) for p in output_dir.rglob('*') if p.is_file() and p not in planned)
    if extra:
        raise ValueError(f'Remove stale files from {output_dir} (sources were renamed or deleted): {extra}')
    return pd.DataFrame(rows)


def safe_name(value):
    """Produce stable, lowercase filesystem-friendly filename components."""
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_') or 'image'


def export_processed_images(source_manifest, output_dir):
    """Save three originals plus eighteen derivatives per eligible person.

    Use the first three valid originals in filename order. Identities with fewer
    originals and explicit unknown folders are skipped. Every output retains its
    original source split; these visualization/export images are never loaded by
    the training pipeline. Reruns overwrite only the same deterministic filenames.
    Images are saved as high-quality JPEG (quality 92) to keep the folder small.
    Existing unexpected JPEGs cause an error instead of deleting user files.
    """
    output_dir = Path(output_dir)
    rows, skipped, plans = [], [], []
    valid = source_manifest[source_manifest.status.eq('ok')]
    for user_id, group in valid.groupby('user_id', sort=True):
        if user_id in UNKNOWN_FOLDERS or len(group) < 3:
            skipped.append(dict(user_id=user_id, original_count=len(group),
                                reason='unknown cohort' if user_id in UNKNOWN_FOLDERS else 'needs at least 3 independent originals'))
            continue
        person_dir = output_dir / safe_name(user_id)
        selected = group.sort_values('path').iloc[:3]
        expected = set()
        person_plan = []
        for number, row in enumerate(selected.itertuples(), start=1):
            stem = safe_name(Path(row.path).stem)
            prefix = f'{safe_name(user_id)}__source_{number:02d}_{stem}__{row.source_id[:12]}'
            for variant in EXPORT_VARIANTS:
                path = person_dir / f'{prefix}__{variant}.jpg'
                expected.add(path.name)
                person_plan.append((row, variant, path))
        if person_dir.exists():
            extra = {p.name for p in person_dir.glob('*.jpg')} - expected
            if extra:
                raise ValueError(f'Archive existing processed images in {person_dir} before changing source images: {sorted(extra)}')
        plans.extend(person_plan)
    for row, variant, path in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Cache one original's variants rather than recomputing for each file.
        if not rows or rows[-1]['source_id'] != row.source_id:
            variants = image_variants(read_rgb(row.path))
        variants[variant].convert('RGB').save(path, format='JPEG', quality=92)
        rows.append(dict(user_id=row.user_id, source_id=row.source_id,
                         source_path=row.path, split=row.split,
                         augmentation=variant, output_path=str(path)))
    exported = pd.DataFrame(rows, columns=['user_id', 'source_id', 'source_path', 'split', 'augmentation', 'output_path'])
    skipped = pd.DataFrame(skipped, columns=['user_id', 'original_count', 'reason'])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_dir.parent / 'processed_image_manifest.csv', index=False)
    skipped.to_csv(output_dir.parent / 'processed_image_skipped.csv', index=False)
    return exported, skipped
