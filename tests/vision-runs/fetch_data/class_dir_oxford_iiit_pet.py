from __future__ import annotations

from _common import copy_image, colors, parse_target_dir, reset_target_dir, split_counts, write_image


def main() -> None:
    target_dir = parse_target_dir("Create a tiny Oxford-IIIT-Pet-style class-dir dataset.")
    reset_target_dir(target_dir)
    palette = colors()
    memberships = (
        ("cat_small", ("cat", "small")),
        ("dog_furry", ("dog", "furry")),
        ("cat_furry", ("cat", "furry")),
        ("dog_small", ("dog", "small")),
    )
    for split, count in split_counts().items():
        for index in range(count):
            stem, labels = memberships[index % len(memberships)]
            base_name = f"{split}_{index:02d}_{stem}.png"
            first_label = labels[0]
            first_path = target_dir / split / first_label / base_name
            write_image(first_path, next(palette))
            for label in labels[1:]:
                copy_image(first_path, target_dir / split / label / base_name)


if __name__ == "__main__":
    main()
