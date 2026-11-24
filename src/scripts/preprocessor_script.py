# preprocess_script.py
import argparse
import os
import tiktoken
import struct

def preprocess_to_bin(input_path, output_dir, batch_size, seq_len, stride):
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    approx_chars_per_token = 5
    read_chars = approx_chars_per_token * (batch_size * seq_len)

    buffer = ""
    file_index = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        while True:
            while len(buffer) < read_chars:
                chunk = f.read(read_chars - len(buffer))
                if not chunk:
                    print()
                    return  # End of file
                buffer += chunk

            tokens = enc.encode(buffer)
            total_needed = batch_size * seq_len
            if len(tokens) < total_needed:
                buffer = buffer[read_chars:]
                continue

            usable_tokens = tokens[:total_needed]

            out_path = os.path.join(output_dir, f"{file_index:08d}.bin")
            with open(out_path, 'wb') as out_file:
                # Write as 4-byte little-endian unsigned ints
                out_file.write(struct.pack(f"<{total_needed}I", *usable_tokens))
            file_index += 1
            print(f"\rBatches written: {file_index}", end="")


            advance_tokens = total_needed - stride
            advance_chars = advance_tokens * approx_chars_per_token
            buffer = buffer[advance_chars:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--stride", type=int, default=0)
    args = parser.parse_args()

    preprocess_to_bin(args.input_path, args.output_dir, args.batch_size, args.seq_len, args.stride)

if __name__ == "__main__":
    main()
