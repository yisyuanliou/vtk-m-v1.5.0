#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

struct Pixel
{
	unsigned char R, G, B; // Blue, Green, Red

	Pixel operator+(const Pixel &p) const
	{
		unsigned char a, b, c;
		a = (p.R + R) > 255 ? 255 : p.R + R;
		b = (p.G + G) > 255 ? 255 : p.G + G;
		c = (p.B + B) > 255 ? 255 : p.B + B;
		// if ((p.R + R) > 255)
		// 	a = 255;
		// else
		// 	a = p.R + R;
		return Pixel{a, b, c};
	}

	Pixel operator*(const float d) const
	{
		unsigned char a, b, c;
		a = (R * d) > 255 ? 255 : R * d;
		b = (G * d) > 255 ? 255 : G * d;
		c = (B * d) > 255 ? 255 : B * d;
		return Pixel{a, b, c};
	}

	Pixel mean(const Pixel &x, const Pixel &y) const
	{
		unsigned char a, b, c;
		a = (x.R + y.R + R) / 3;
		b = (x.G + y.G + G) / 3;
		c = (x.B + y.B + B) / 3;
		return Pixel{a, b, c};
	}
};

class ColorImage
{
	Pixel *pPixel;
	int xRes, yRes;

public:
	ColorImage();
	~ColorImage();
	void init(int xSize, int ySize);
	void clear(Pixel background);
	Pixel readPixel(int x, int y);
	void writePixel(int x, int y, Pixel p);
	void outputPPM(char *filename);
};

ColorImage::ColorImage()
{
	pPixel = 0;
}

ColorImage::~ColorImage()
{
	if (pPixel)
		delete[] pPixel;
	pPixel = 0;
}

void ColorImage::init(int xSize, int ySize)
{
	Pixel p = {0, 0, 0};
	xRes = xSize;
	yRes = ySize;
	pPixel = new Pixel[xSize * ySize];
	clear(p);
}

void ColorImage::clear(Pixel background)
{
	int i;

	if (!pPixel)
		return;
	for (i = 0; i < xRes * yRes; i++)
		pPixel[i] = background;
}

Pixel ColorImage::readPixel(int x, int y)
{
	assert(pPixel); // die if image not initialized
	return pPixel[x + y * yRes];
}

void ColorImage::writePixel(int x, int y, Pixel p)
{
	assert(pPixel); // die if image not initialized
	pPixel[x + y * yRes] = p;
}

void ColorImage::outputPPM(char *filename)
{
	FILE *outFile = fopen(filename, "wb");

	assert(outFile); // die if file can't be opened

	fprintf(outFile, "P6 %d %d 255\n", xRes, yRes);
	fwrite(pPixel, 1, 3 * xRes * yRes, outFile);

	fclose(outFile);
}
