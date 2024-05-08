function [EO]=loggabor(im)

    [rows,cols] = size(im(:,:,1));
    I1 = ones(rows, cols);
    Q1 = ones(rows, cols);

    if ndims(im) == 3 %images are colorful
        Y1=0.299 * double(im(:,:,1)) + 0.587 * double(im(:,:,2))+ 0.114 * double(im(:,:,3));
        I1 = 0.596 * double(im(:,:,1)) - 0.274 * double(im(:,:,2)) - 0.322 *double(im(:,:,3));
        Q1 = 0.211 *double(im(:,:,1)) - 0.523 * double(im(:,:,2)) + 0.312 * double(im(:,:,3));
    else
       Y1 = im;
    end
    Y1 = double(Y1);

    minDimension = min(rows,cols);
    F = max(1,round(minDimension / 256));

    aveKernel = fspecial('average',F);
    aveI1 = conv2(I1, aveKernel,'same');
    I1 = aveI1(1:F:rows,1:F:cols);
    aveQ1 = conv2(Q1, aveKernel,'same');
    Q1 = aveQ1(1:F:rows,1:F:cols);
    aveY1 = conv2(Y1, aveKernel,'same');
    Y1 = aveY1(1:F:rows,1:F:cols);

    nscale = 5;
    norient = 4;
    minWaveLength = 6;
    mult = 2;
    sigmaOnf = 0.55;
    dThetaOnSigma = 1.2;
    k = 2.0;
    epsilon = .0001;
    thetaSigma = pi/norient/dThetaOnSigma;
    [rows,cols] = size(Y1);


    imagefft = fft2(Y1);
    zero = zeros(rows,cols);
    EO = cell(nscale, norient);
    estMeanE2n = [];
    ifftFilterArray = cell(1,nscale);
    if mod(cols,2)
        xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
    else xrange = [-cols/2:(cols/2-1)]/cols;
    end
    if mod(rows,2)
        yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
    else yrange = [-rows/2:(rows/2-1)]/rows;
    end

    [x,y] = meshgrid(xrange, yrange);
    radius = sqrt(x.^2 + y.^2);
    theta = atan2(-y,x);

    radius = ifftshift(radius);
    theta = ifftshift(theta);
    radius(1,1) = 1;
    sintheta = sin(theta);
    costheta = cos(theta);
    clear x;
    clear y;
    clear theta;

    lp = lowpassfilter([rows,cols],.45,15);
    logGabor = cell(1,nscale);
    for s = 1:nscale
        wavelength = minWaveLength*mult^(s-1);
        fo = 1.0/wavelength;
        logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));
        logGabor{s} = logGabor{s}.*lp;
        logGabor{s}(1,1) = 0;

    end
    spread = cell(1,norient);
    for o = 1:norient
        angl = (o-1)*pi/norient;
        ds = sintheta * cos(angl) - costheta * sin(angl);
        dc = costheta * cos(angl) + sintheta * sin(angl);
        dtheta = abs(atan2(ds,dc));
        spread{o} = exp((-dtheta.^2) / (2 * thetaSigma^2));
    end % The main loop...
    EnergyAll(rows,cols) = 0;
    AnAll(rows,cols) = 0;


    for o = 1:norient
        sumE_ThisOrient = zero;
        sumO_ThisOrient = zero;
        sumAn_ThisOrient = zero;
        Energy = zero;
        for s =1:nscale
            filter = logGabor{s} .* spread{o};
            ifftFilt = real(ifft2(filter))*sqrt(rows*cols);
            ifftFilterArray{s} = ifftFilt;
            EO{s,o} = ifft2(imagefft .* filter);
        end
    end
