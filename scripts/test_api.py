import argparse
import mimetypes
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post_image(url, image_path):
    boundary = "----PlantDiseaseApiTestBoundary"
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    image_data = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()

    status, response_body = post_image(args.url, args.image)
    print(f"HTTP status: {status}")
    if status >= 400:
        raise RuntimeError(response_body.decode())
    import json

    result = json.loads(response_body)
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Class index: {result['class_index']}")


if __name__ == "__main__":
    main()