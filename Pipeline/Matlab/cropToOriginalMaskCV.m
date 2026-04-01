function cropToOriginalMaskCV(input_maskOriginal, input_maskAC, output_maskAC, coordinatesMin, coordinatesMax)
    % Read the images
    try
        original_image = imread(input_maskOriginal);
    catch e
        disp(['Error reading original image: ' e.message]);
        return;
    end
    
    try
        crop_binary_mask = imread(input_maskAC);
    catch e
        disp(['Error reading crop mask: ' e.message]);
        return;
    end

    % Get the original image dimensions
    [original_height, original_width, ~] = size(original_image);
    
    % Create a black mask with the same size as the original image
    black_mask = zeros(original_height, original_width, 'uint8');

    % Define the coordinates of crop in the original image
    xmin = max(1, round(coordinatesMin{1}));
    ymin = max(1, round(coordinatesMin{2}));
    xmax = min(original_width, round(coordinatesMax{1}));
    ymax = min(original_height, round(coordinatesMax{2}));
    
    % Verify that the dimensions are valid
    if xmin >= xmax || ymin >= ymax
        disp('Error: Dimensions of crop are not valid');
        disp(['xmin: ' num2str(xmin) ', xmax: ' num2str(xmax) ', ymin: ' num2str(ymin) ', ymax: ' num2str(ymax)]);
        return;
    end
    
    % Obtain the dimensions of the crop mask
    [crop_height, crop_width] = size(crop_binary_mask);
    
    % Verify if the dimension of the crop mask coincide with the crop area
    expected_width = xmax - xmin;
    expected_height = ymax - ymin;
    
    % Ajust the crop mask if necessary
    if crop_width ~= expected_width || crop_height ~= expected_height
        disp('Ajusting dimensions of the crop mask...');
        crop_binary_mask = imresize(crop_binary_mask, [expected_height, expected_width]);
    end
    
    % Put the crop binary mask in the corresponding coordinates of the black mask
    try
        black_mask(ymin:ymax-1, xmin:xmax-1) = crop_binary_mask;
    catch e
        disp(['Error inserting mask: ' e.message]);
        disp(['Dimensions of the crop mask: ' num2str(size(crop_binary_mask))]);
        disp(['Range of insertion: y=' num2str(ymin) ':' num2str(ymax-1) ', x=' num2str(xmin) ':' num2str(xmax-1)]);
        return;
    end

    % Save the binary ajusted mask
    try
        imwrite(black_mask, output_maskAC);
        disp('CV whole mask completed');
    catch e
        disp(['Error saving ajusted mask: ' e.message]);
    end
end
