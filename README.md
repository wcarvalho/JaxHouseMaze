# housemaze

This is a simple maze environment where you can easily describe maps with strings. The map can either be symbolic, where each tile is represented by the category at that time, or it can be visual, where images for categories can be loaded in via a `image_data.pkl` file.

For example, the following string and dicionary:
```
maze = """
.#.C...##....
.#..D...####.
.######......
......######.
.#.#..#......
.#.#.##..#...
##.#.#>.###.#
A..#.##..#...
.B.#.........
#####.#..####
......####.#.
.######E.#.#.
........F#...

char_to_key=dict(
    A="knife",
    B="fork",
    C="pan",
    D="pot",
    E="bowl",
    F="plates",
)
"""
````
can render the following image:

<img src="example.png" alt="FARM" style="zoom:40%;" />

Images for categories are loaded from a `image_data.pkl` file that needs to have the following two fields:
- `keys`
- `images`

Please see this [jupyter notebook](exmaple.ipdb) for an interactive example loading the environment.
