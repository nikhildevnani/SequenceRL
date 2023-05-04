import os

from PIL import Image, ImageDraw, ImageFont

from helper_functions import get_cards_name_positions, get_card_name_to_number_mapping

CORNER_LOCATIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]

# Create a new image with a white background
width, height = 1000, 1000
RED = 255, 0, 0
BLUE = 0, 0, 255
WHITE = 255, 255, 255

# Set the font and size
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
font_path = os.path.join(current_dir, 'lato', 'lato-black.ttf')
font = ImageFont.truetype(font_path, 12)
card_names_on_positions = get_cards_name_positions()
card_numbers_from_names = get_card_name_to_number_mapping()


def generate_image(matrix, image_name, actual_hand_cards):
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # Set the size of each grid cell
    cell_size = width // 10
    m = len(matrix)
    n = len(matrix[0])

    # Draw the text with the background color for each grid cell
    for i in range(m):
        for j in range(n):
            location = (i, j)
            text_color = 'black'
            if location not in CORNER_LOCATIONS:
                card_name = card_names_on_positions[(i, j)]
                card_number = card_numbers_from_names[card_name]
                if card_number in actual_hand_cards:
                    text_color = 'green'
                text = f"({i}, {j}, {card_name})"
            else:
                text = "corner"

            x = j * cell_size
            y = i * cell_size
            color = WHITE
            if matrix[location] == 0:
                color = RED
            if matrix[location] == 1:
                color = BLUE
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color)
            draw.text((x + cell_size // 2, y + cell_size // 2), text, fill=text_color, font=font, anchor='mm')

    # Save the image
    image.save(image_name)
