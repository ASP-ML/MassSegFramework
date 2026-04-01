function filtered_oneImage(input, output)
    grayImage = imread(input);
    paddingSize = floor([5 5] / 2);
    paddedImage = padarray(grayImage, paddingSize, 'replicate');

    % Apply median filter
    filteredImage = medfilt2(paddedImage, [5 5]);
    filteredImage = filteredImage(paddingSize(1)+1:end-paddingSize(1), paddingSize(2)+1:end-paddingSize(2));

    [rows, cols] = size(filteredImage);
    rgbImage = cat(3, filteredImage, filteredImage, filteredImage);

    imwrite(rgbImage, output);

    disp('Filtered completed');
end