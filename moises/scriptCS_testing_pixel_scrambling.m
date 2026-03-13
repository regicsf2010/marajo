%% Initialization
clc; clear; close all;

%% Load Video

tic

vidReader = VideoReader('Video 1 raw_400.avi');

nFrames = round(vidReader.Duration*vidReader.FrameRate);

Fs = vidReader.FrameRate;
nRows = vidReader.Height;
nColumns = vidReader.Width;
dataset = zeros(nFrames,nRows*nColumns);
Frames = zeros(nRows,nColumns,nFrames);

i = 1;
while hasFrame(vidReader)
    frameGray = rgb2gray(readFrame(vidReader));
    Frames(:,:,i) = frameGray;
    dataset(i,:) = frameGray(:);
    i = i+1;
end

M = nRows*nColumns;

% RANDOM PIXEL SCRAMBLING (ENCRYPTION KEY)
ridx = randperm(M);
dataset = dataset(:,ridx); % ENCRYPTION KEY

% VECTOR WITH THE CORRECT PIXEL ORDER (DECRYPTION KEY)
sidx=1:length(ridx);
sidx=[ridx' sidx'];
sidx=sortrows(sidx,1);

sidx = sidx(:,2); % DECRYPTION KEY

% visualizing original (unscrambled) and scrambled videos
for i=1:nFrames
    Frames2(:,:,i) = [reshape(dataset(i,sidx),nRows,nColumns) ...
                      reshape(dataset(i,:),nRows,nColumns)];
end

implay(uint8(Frames2),100)




minVal = min(Frames(:)); maxVal = max(Frames(:));

clear s vidReader frameGray

dt = 1/Fs;
freq = (0:nFrames-1)'/(nFrames/Fs);
t = (0:nFrames-1)'/Fs;

% background equalization (only dynamic changes are retained)
Mean = mean(dataset);
dataset = dataset-Mean;

clear Frames2

%% Dataset augmentation

disp('Dataset augmentation')

% real data (original)
X0 = dataset;

% Hilbert transform: apply 90° rotation to discard negative freq components
H = hilbert(X0);

% complex-part: 90° phase shifted data 
X90 = imag(H);

disp('End of augmentation')

clear H dataset

%% PCA

disp('Dim. Red.: PCA')

% PCA over real-part
disp('PCA over X0')
[H0,W0,V0] = pca(X0');


% PCA over imag-part
disp('PCA over X90')
[H90,W90,V90] = pca(X90');

disp('End of Dim. Red.')

%% Sorting real and imaginary components

V = [V0; V90];

[V,idx]=sort(V,'desc');

H = [H0 H90];
W = [W0 W90];

H = H(:,idx);
W = W(:,idx);

clear H0 H90 W0 W90

%% Blind source separation (BSS) based on complexity pursuit (CP)

numPC = 16; % use the first 16 principal components for BSS

disp('Blind Source Separation')

% BSS
[unmixed,Wmix] = CP_alg(H(:,1:numPC));
% Winvmix = inv(Wmix);
Winvmix = flip(inv(Wmix));
unmixed = -fliplr(unmixed);

disp('End of Blind Source Separation')


%% Visualizing the sources

% sources
h = figure;
j=[1 2 3];
for i=1:numPC
    subplot(numPC,3,j(1))
    plot(t,unmixed(:,i),'linewidth',1.5)
    title(['Source ' num2str(i)])
    set(gca,'fontsize',9)
    box on
    
    subplot(numPC,3,j(2))
    aux = abs(fft(unmixed(:,i))).^2;
    plot(freq(2:round(nFrames/2)),aux(2:round(nFrames/2)),'linewidth',1.5,'color','k')
    if i==1
        title('PSD')
    end
    set(gca,'fontsize',9)
    box on
    
    subplot(numPC,3,j(3))
    aux = angle(fft(unmixed(:,i))).^2;
    plot(freq(2:round(nFrames/2)),aux(2:round(nFrames/2)),'linewidth',1.5,'color','k')
    if i==1
        title('Phase')
    end
    set(gca,'fontsize',9)
    box on
        
    j=j+3;
    
end
subplot(numPC,3,j(1)-3)
xlabel('Time (s)')
subplot(numPC,3,j(2)-3)
xlabel('Frequency (Hz)')
set(h,'position',[150 150 700 550]);

%% Solving the mode shapes and modal coordinates

% from the 16 sources only the ones exhibiting a monotone behavior are useful 
% here, we manually specify which ones are important
srcs = [1 2 9 10 14 15];

% number of real and imaginary modes
numSrc = length(srcs);

% note that the sources are already our modal coordinates
modal_coord = -unmixed(:,srcs);

% for the mode shapes we still need to solve the uncoupled mode equation
mode_shapes = (Winvmix*W(:,1:numPC)')';
mode_shapes = mode_shapes(:,srcs);


%% Visualizing the scrambled mode shapes

h = figure;
j=1;
for i=1:numSrc
    subplot(2,numSrc/2,j)
    S = mode_shapes(:,i)'; 
    imagesc(reshape(S,nRows,nColumns))
    title(['Scrambled Mode Shape ' num2str(i)],'fontsize',12)
    set(gca,'visible','off')
    set(get(gca,'Title'),'Visible','on')
    set(h,'position',[150 150 300 550]);
    colorbar
    j=j+1;
end
set(h,'position',[150 150 700 550]);

%% Visualizing the unscrambled mode shapes

h = figure;
j=1;
for i=1:numSrc
    subplot(2,numSrc/2,j)
    S = mode_shapes(sidx,i)'; 
    imagesc(reshape(S,nRows,nColumns))
    title(['Unscrambled Mode Shape ' num2str(i)],'fontsize',12)
    set(gca,'visible','off')
    set(get(gca,'Title'),'Visible','on')
    set(h,'position',[150 150 300 550]);
    colorbar
    j=j+1;
end
set(h,'position',[150 150 700 550]);


%% Visualizing the modal coordinates

h = figure;
j=[1 2 3];
for i=1:numSrc
    subplot(numSrc,3,j(1))
    plot(t,modal_coord(:,i),'linewidth',1.5)
    title(['Coordinate ' num2str(i)])
    set(gca,'fontsize',9)
    box on
    
    subplot(numSrc,3,j(2))
    aux = abs(fft(modal_coord(:,i))).^2;
    plot(freq(2:round(nFrames/2)),aux(2:round(nFrames/2)),'linewidth',1.5,'color','k')
    if i==1
        title('PSD')
    end
    set(gca,'fontsize',9)
    box on
    
    subplot(numSrc,3,j(3))
    aux = angle(fft(modal_coord(:,i))).^2;
    plot(freq(2:round(nFrames/2)),aux(2:round(nFrames/2)),'linewidth',1.5,'color','k')
    if i==1
        title('Phase')
    end
    set(gca,'fontsize',9)
    box on
        
    j=j+3;
    
end
subplot(numSrc,3,j(1)-3)
xlabel('Time (s)')
subplot(numSrc,3,j(2)-3)
xlabel('Frequency (Hz)')
set(h,'position',[150 150 700 550]);


%% Reconstructing the original video from the scrambled shapes and coordinates
% also, each mode has its motion magnified for visualization

clear Frames2 datasetrec1 datasetrec2 datasetrec3 datasetrec4
disp('Reconstructing sources')

% reconstructing the whole video from these 6 components
src0 = modal_coord*mode_shapes';

% reconstructing videos for each mode after motion magnification

beta1 = 5; beta2 = 15; beta3 = 30; % arbitrary motion amplification parameter

src1 = beta1*modal_coord(:,[1 2])*mode_shapes(:,[1 2])' ...
       -modal_coord(:,[3 4])*mode_shapes(:,[3 4])'...
       -modal_coord(:,[5 6])*mode_shapes(:,[5 6])';
   
src2 = -modal_coord(:,[1 2])*mode_shapes(:,[1 2])' ...
       +beta2*modal_coord(:,[3 4])*mode_shapes(:,[3 4])'...
       -modal_coord(:,[5 6])*mode_shapes(:,[5 6])';
   
src3 = -modal_coord(:,[1 2])*mode_shapes(:,[1 2])' ...
       -modal_coord(:,[3 4])*mode_shapes(:,[3 4])'...
       +beta3*modal_coord(:,[5 6])*mode_shapes(:,[5 6])';   


disp('End sources reconstruction')

% inserting 
bckgd = double(reshape(Mean,nRows,nColumns));

% reshaping, rescaling, and re-building the frames
for j=1:nFrames

    F0 = src0(j,:);
    F0 = reshape(F0,nRows,nColumns)+bckgd;
    
    F1 = src1(j,:);
    F1 = reshape(F1,nRows,nColumns)+bckgd;
    
    F2 = src2(j,:);
    F2 = reshape(F2,nRows,nColumns)+bckgd;
    
    F3 = src3(j,:);
    F3 = reshape(F3,nRows,nColumns)+bckgd;
    

    
    Frames2(:,:,j) = uint8([Frames(:,:,j)     ...
                            255*ones(nRows,2) ...
                            F0         ...
                            255*ones(nRows,2) ...
                            F1                ...
                            255*ones(nRows,2) ...
                            F2                ...
                            255*ones(nRows,2) ...
                            F3 ]);
   
end

implay(Frames2,50)

%% Reconstructing the original video from the unscrambled shapes and coordinates
% also, each mode has its motion magnified for visualization

clear Frames2 datasetrec1 datasetrec2 datasetrec3 datasetrec4
disp('Reconstructing sources')

% reconstructing the whole video from these 6 components
src0 = modal_coord*mode_shapes';

% reconstructing videos for each mode after motion magnification

beta1 = 5; beta2 = 15; beta3 = 30; % arbitrary motion amplification parameter

src1 = beta1*modal_coord(:,[1 2])*mode_shapes(:,[1 2])' ...
       -modal_coord(:,[3 4])*mode_shapes(:,[3 4])'...
       -modal_coord(:,[5 6])*mode_shapes(:,[5 6])';
   
src2 = -modal_coord(:,[1 2])*mode_shapes(:,[1 2])' ...
       +beta2*modal_coord(:,[3 4])*mode_shapes(:,[3 4])'...
       -modal_coord(:,[5 6])*mode_shapes(:,[5 6])';
   
src3 = -modal_coord(:,[1 2])*mode_shapes(:,[1 2])' ...
       -modal_coord(:,[3 4])*mode_shapes(:,[3 4])'...
       +beta3*modal_coord(:,[5 6])*mode_shapes(:,[5 6])';   


disp('End sources reconstruction')

% inserting 
bckgd = double(reshape(Mean(sidx),nRows,nColumns));
Error = zeros(nRows,nColumns);

% reshaping, rescaling, re-building and unscrambling the frames
for j=1:nFrames

    F0 = src0(j,sidx);
    F0 = reshape(F0,nRows,nColumns)+bckgd;
    
    F1 = src1(j,sidx);
    F1 = reshape(F1,nRows,nColumns)+bckgd;
    
    F2 = src2(j,sidx);
    F2 = reshape(F2,nRows,nColumns)+bckgd;
    
    F3 = src3(j,sidx);
    F3 = reshape(F3,nRows,nColumns)+bckgd;
    

    
    Frames2(:,:,j) = uint8([Frames(:,:,j)     ...
                            255*ones(nRows,2) ...
                            F0         ...
                            255*ones(nRows,2) ...
                            F1                ...
                            255*ones(nRows,2) ...
                            F2                ...
                            255*ones(nRows,2) ...
                            F3 ]);
                      
	Error = Error+(Frames(:,:,j)-F0);
end

% norm between the original video and the reconstructed one
Norm = sum(Error(:)).^2

% evaluating the rate of compression if transmitting only the mode_shapes
% and modal_coordinates instead of the whole video
whos ('sidx','mode_shapes', 'modal_coord', 'Frames')
[obj]=whos ('sidx','mode_shapes', 'modal_coord', 'Frames')
% obj.bytes

% this time we are also taking into account the size of the decryption key
compression_rate = 1-(663552+19200+3981312)/265420800

implay(Frames2,50)

toc
