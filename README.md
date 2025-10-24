# XnYZip

This is the XnYZip project.

> The expected platform is Linux. Haven't tested on other platforms.

# Building and installing

See the [HACKING](HACKING.md) and [BUILDING](BUILDING.md) document.

# Usage

## Data Preparation

Download by `https://www.dropbox.com/scl/fi/1117g2niw8w0gjrffc8l3/Point-cloud-data.zip?dl=1`. The Viewer info has been disabled for double-blind requirement.

After downloading (the file name should be `Point-cloud-data.zip`), use `unzip Point-cloud-data.zip` to decompress.

There should be two files: `usgs.bin` and `vesicles_1_75M.bin`.

## Run

After build XnYZip:

```sh
./build/XnYZip <input_file> <quantizer_type (cube/to)> <L2 bound> <-z/-h> <-rle/-normal> <decompression_file_name>
```

+ `-z` or `-h` here means using z-order curve or hilbert curve
+ use `-rle` and `-normal` to control the usage of RLE

input file should be a float32 bin file, arranged like x1, y1, z1, x2, y2, z2 .... The provided bin files is in this format already.

# Licensing

<!--
Please go to https://choosealicense.com/licenses/ and choose a license that
fits your needs. The recommended license for a project of this type is the
GNU AGPLv3.
-->
