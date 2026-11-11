"""Save an inspectable 3-original to 21-image dataset with source provenance."""
from pathlib import Path
import re

import pandas as pd

from .embeddings import read_rgb
from .pipeline import UNKNOWN_FOLDERS, image_variants

EXPORT_VARIANTS = ('original', 'rotation_plus_8', 'horizontal_flip', 'grayscale',
                   'clahe', 'gamma_080', 'median_blur_3')


def safe_name(value):
    """Produce stable, lowercase filesystem-friendly filename components."""
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_') or 'image'


def export_processed_images(source_manifest, output_dir):
    """Save three originals plus eighteen derivatives per eligible person.

    Use the first three valid originals in filename order. Identities with fewer
    originals and explicit unknown folders are skipped. Every output retains its
    original source split; these visualization/export images are never loaded by
    the training pipeline. Reruns overwrite only the same deterministic filenames.
    Existing unexpected PNGs cause an error instead of deleting user files.
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
                path = person_dir / f'{prefix}__{variant}.png'
                expected.add(path.name)
                person_plan.append((row, variant, path))
        if person_dir.exists():
            extra = {p.name for p in person_dir.glob('*.png')} - expected
            if extra:
                raise ValueError(f'Archive existing processed images in {person_dir} before changing source images: {sorted(extra)}')
        plans.extend(person_plan)
    for row, variant, path in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Cache one original's variants rather than recomputing for each file.
        if not rows or rows[-1]['source_id'] != row.source_id:
            variants = image_variants(read_rgb(row.path))
        variants[variant].save(path, format='PNG')
        rows.append(dict(user_id=row.user_id, source_id=row.source_id,
                         source_path=row.path, split=row.split,
                         augmentation=variant, output_path=str(path)))
    exported = pd.DataFrame(rows, columns=['user_id', 'source_id', 'source_path', 'split', 'augmentation', 'output_path'])
    skipped = pd.DataFrame(skipped, columns=['user_id', 'original_count', 'reason'])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_dir.parent / 'processed_image_manifest.csv', index=False)
    skipped.to_csv(output_dir.parent / 'processed_image_skipped.csv', index=False)
    return exported, skipped
