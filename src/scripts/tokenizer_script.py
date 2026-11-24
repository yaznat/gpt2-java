import sys
import tiktoken # TikToken needs to be installed to a virtual environment
import base64

enc = tiktoken.get_encoding("gpt2")

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line.startswith("DETOKENIZE "):
            token_str = line[len("DETOKENIZE "):]
            tokens = [int(t) for t in token_str.split() if t.strip()]
            decoded_text = decode(tokens)
            print(decoded_text)
            print("__END__", flush=True)
        else:
            # Decode base64 text input
            try:
                text_bytes = base64.b64decode(line)
                text = text_bytes.decode("utf-8")
                tokens = enc.encode(text, allowed_special={"<|endoftext|>"}, disallowed_special=())
                print(" ".join(str(t) for t in tokens), flush=True)
            except Exception as e:
                print(f"ERROR decoding base64: {e}", flush=True)

if __name__ == "__main__":
    main()