"""
Demonstration of using some of the pyvision tools to
draw polygons on an image and record the shapes,
to crop tiles around the polygons, and use a
tile selector to "QA" the tiles.
"""
import pyvision3 as pv3
import random


def demo():
    print("=================================================================")
    print("STEP 1: Draw two or more polygons on the sleepy cat image...")
    print(" With image in focus, hit space-bar when done.")
    print("=================================================================")
    img = pv3.Image(pv3.IMG_SLEEPYCAT)
    cp = pv3.CapturePolygons(img)
    cp.display()
    poly_list = cp.current_polygons

    print("=================================================================")
    print("STEP 2: We've cropped a rectangle around each polygon, and we")
    print(" mixed in 15 random tiles from the area outside the polygons.")
    print("=================================================================")

    # crop rectangular tiles around each drawn polygon
    tiles = pv3.crop_regions(img, poly_list)
    num_tiles_created = len(tiles)
    your_tile_ids = [str(i) for i in range(num_tiles_created)]

    # generate random 'negative' crops
    tiles2 = pv3.crop_negative_regions(img, poly_list, crop_size=(150, 150), N=15)

    # combine drawn tiles and random tiles and shuffle together
    all_tiles = [(idx, tile) for (idx, tile) in enumerate(tiles + tiles2)]
    random.shuffle(all_tiles)  # modifies list in-place

    print("=================================================================")
    print("STEP 3: Use tile selector to select the tiles you created, ")
    print(" not the random crops...if you can! After selecting the tiles")
    print(" on a page, hit the space bar to go to the next page.")
    print("=================================================================")
    tile_generator = (
        (str(id), tile, "label_{}".format(id)) for (id, tile) in all_tiles
    )
    ts = pv3.TileSelector(
        tile_generator, chunk_size=8, layout=(2, 4), tile_size=(150, 150)
    )
    ts.process_all()
    selected_tiles = sorted(ts.selected)

    print("=================================================================")
    print("RESULTS:")
    print("=================================================================")
    print("You selected the following tiles: {}".format(selected_tiles))
    print("You should have selected: {}".format(your_tile_ids))
    if selected_tiles == your_tile_ids:
        print("WE HAVE A WINNER!")


if __name__ == "__main__":
    demo()
