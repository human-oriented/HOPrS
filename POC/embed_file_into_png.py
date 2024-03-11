import argparse
import struct
import zlib

def create_chunk(chunk_type, data):
    """Create a PNG chunk with given type and data, including length and CRC."""
    chunk_len = len(data)
    chunk_data = chunk_type + data
    crc = zlib.crc32(chunk_data)
    return struct.pack("!I", chunk_len) + chunk_data + struct.pack("!I", crc)

def embed_text_in_png(png_path, text_path, output_png_path):
    """Embeds text from a file into a PNG image as a custom chunk."""
    with open(text_path, 'r', encoding='utf-8') as file:
        text_data = file.read().encode('utf-8')

    # Define custom chunk type (4 bytes)
    chunk_type = b'txTe'

    # Read the original PNG data
    with open(png_path, 'rb') as file:
        original_data = file.read()

    # Find the position of the IEND chunk
    iend_position = original_data.rfind(b'IEND')

    # Create the custom chunk
    custom_chunk = create_chunk(chunk_type, text_data)

    # Insert the custom chunk before IEND
    new_png_data = original_data[:iend_position-4] + custom_chunk + original_data[iend_position-4:]

    # Write the modified data to a new file
    with open(output_png_path, 'wb') as file:
        file.write(new_png_data)

    print(f"Text from '{text_path}' embedded into '{output_png_path}'")

def extract_text_from_png(png_path):
    """Extracts text embedded in a PNG image as a custom chunk."""
    with open(png_path, 'rb') as file:
        data = file.read()

    start = 8  # Skip the PNG signature
    found_text = False

    while start < len(data):
        chunk_len = struct.unpack('!I', data[start:start+4])[0]
        chunk_type = data[start+4:start+8]
        chunk_data = data[start+8:start+8+chunk_len]
        # Move to the next chunk (length + type + data + CRC)
        start += chunk_len + 12

        if chunk_type == b'txTe':
            text_data = chunk_data.decode('utf-8')
            print("Extracted text:", text_data)
            found_text = True
            break

    if not found_text:
        print("No embedded text found.")

def main():
    parser = argparse.ArgumentParser(description="Embed or extract text in/from a PNG file.")
    subparsers = parser.add_subparsers(dest="command")

    embed_parser = subparsers.add_parser("embed", help="Embed text into a PNG file")
    embed_parser.add_argument("png_path", help="Path to the source PNG file")
    embed_parser.add_argument("text_path", help="Path to the text file to embed")
    embed_parser.add_argument("output_png_path", help="Path to the output PNG file")

    extract_parser = subparsers.add_parser("extract", help="Extract text from a PNG file")
    extract_parser.add_argument("png_path", help="Path to the PNG file with embedded text")

    args = parser.parse_args()

    if args.command == "embed":
        embed_text_in_png(args.png_path, args.text_path, args.output_png_path)
    elif args.command == "extract":
        extract_text_from_png(args.png_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
