from PIL import Image, ImageDraw, ImageFont


def matrix_to_image(matrix, text_colors, bg_colors, cell_width=50, cell_height=50, font_size=20):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    image_width = num_cols * cell_width
    image_height = num_rows * cell_height

    # create a new image
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # set up the font
    font = ImageFont.truetype('lato/lato-black.ttf', font_size)

    # iterate over each cell in the matrix
    for row in range(num_rows):
        for col in range(num_cols):
            # get the text and background color for this cell
            text = matrix[row][col]
            text_color = text_colors[row][col]
            bg_color = bg_colors[row][col]

            # calculate the position of the cell in the image
            x = col * cell_width
            y = row * cell_height

            # draw the cell
            draw.rectangle((x, y, x + cell_width, y + cell_height), fill=bg_color)
            draw.text((x + cell_width // 2, y + cell_height // 2), text, fill=text_color, font=font, anchor='mm')

    return image


matrix_to_image([[1, 0, 1]], [['red', 'blue', 'red']], [['white', 'white', 'white']])
