# XnYZip

This is the XnYZip project.

# Building and installing

See the [HACKING](HACKING.md) and [BUILDING](BUILDING.md) document.

# Usage

After build XnYZip:

```sh
./build/XnYZip <input_file> <quantizer_type (cube/to)> <L2 bound> <-z/-h> <-rle/-normal> <decompression_file_name>
```

+ `-z` or `-h` here means using z-order curve or hilbert curve
+ use `-rle` and `-normal` to control the usage of RLE

input file should be a float32 bin file, arranged like x1, y1, z1, x2, y2, z2 ....


# Licensing

<!--
Please go to https://choosealicense.com/licenses/ and choose a license that
fits your needs. The recommended license for a project of this type is the
GNU AGPLv3.
-->
