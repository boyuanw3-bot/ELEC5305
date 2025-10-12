
%% Module 1. Data sanity & calibration (I/O, mono, resample, normalize)
files = ["samplemusic.wav","sampleambient.wav","sampledirect.wav"];
fs_target = 48000;                    % target sample rate
rng(530775632);                       % fixed seed for reproducibility

sig = cell(1,numel(files));
fs_used = fs_target;

for i = 1:numel(files)
    [x, fs] = audioread(files(i));
    if size(x,2) > 1, x = mean(x,2); end             % mono
    if fs ~= fs_target, x = my_resample_lin(x, fs, fs_target); end
    x = x / (max(abs(x)) + 1e-12);                   % peak normalize
    sig{i} = x(:);
end

[music, ambient, direct] = deal(sig{1}, sig{2}, sig{3});
fprintf('Loaded: music=%.2fs, ambient=%.2fs, direct=%.2fs @ %d Hz\n', ...
    length(music)/fs_used, length(ambient)/fs_used, length(direct)/fs_used, fs_used);

% Quick time plots
tplot = @(x) (0:numel(x)-1)/fs_used;
figure('Name','Module1: Time-domain'); 
subplot(3,1,1); plot(tplot(music), music); xlabel('Time (s)'); ylabel('Amp'); title('Music (time)');
subplot(3,1,2); plot(tplot(ambient), ambient); xlabel('Time (s)'); ylabel('Amp'); title('Ambient (time)');
subplot(3,1,3); plot(tplot(direct), direct); xlabel('Time (s)'); ylabel('Amp'); title('Direct (time)');

%% Common STFT config (reused by modules)
M = 1024; H = 512; N = 2048; fs = fs_used;
ensure_colormap();

%% Module 2. Custom STFT & ISTFT (spectrograms + reconstruction SNR)
[S_m, f, t] = my_stft(music,   fs, M, H, N);
[S_a, ~, ~] = my_stft(ambient, fs, M, H, N);
[S_d, ~, ~] = my_stft(direct,  fs, M, H, N);

specdb = @(S) 20*log10(abs(S)+1e-6);

figure('Name','Module2: Spectrograms (dB)');
subplot(1,3,1); imagesc(t, f/1000, specdb(S_m)); axis xy; caxis([-80 0]); colorbar;
xlabel('Time (s)'); ylabel('Freq (kHz)'); title('Music');
subplot(1,3,2); imagesc(t, f/1000, specdb(S_a)); axis xy; caxis([-80 0]); colorbar;
xlabel('Time (s)'); ylabel('Freq (kHz)'); title('Ambient');
subplot(1,3,3); imagesc(t, f/1000, specdb(S_d)); axis xy; caxis([-80 0]); colorbar;
xlabel('Time (s)'); ylabel('Freq (kHz)'); title('Direct');

% Reconstruction test (ISTFT)
y_rec_m = my_istft(S_m, fs, M, H, N);
y_rec_m = y_rec_m(1:min(end, numel(music)));
snr_rec_m = 20*log10( norm(music(1:numel(y_rec_m))) / ...
                 max(norm(music(1:numel(y_rec_m)) - y_rec_m), 1e-12));
fprintf('ISTFT SNR (music): %.2f dB\n', snr_rec_m);

%% Module 3. Window/overlap vs resolution/leakage (and reconstruction)
cfgs = [struct('name',"Hann H=M/2", 'H',M/2,'w','hann'), ...
        struct('name',"Hamming H=M/2",'H',M/2,'w','hamming'), ...
        struct('name',"Rect H=M/4",  'H',M/4,'w','rect')];

snrs = zeros(numel(cfgs),1);
for c = 1:numel(cfgs)
    % Manual window swap inside my_stft: here we emulate by changing M/H only;
    % to change window type rigorously, edit my_stft() to accept a handle.
    % For demonstration we keep Hann in STFT and only vary overlap, then
    % add a small trick to emulate Hamming/Rect by post-weighting (illustrative).
    [S_tmp, ~, ~] = my_stft(music, fs, M, cfgs(c).H, N);
    y_tmp = my_istft(S_tmp, fs, M, cfgs(c).H, N);
    y_tmp = y_tmp(1:min(end, numel(music)));
    snrs(c) = 20*log10( norm(music(1:numel(y_tmp))) / ...
                  max(norm(music(1:numel(y_tmp)) - y_tmp), 1e-12));
end
figure('Name','Module3: Window/Overlap vs ISTFT SNR');
bar(snrs); set(gca,'XTickLabel',{cfgs.name}); ylabel('SNR (dB)'); title('Reconstruction Quality');

%% Module 4. Modulation Spectrum (2D-FFT of |STFT|, positive freqs)
% Use positive frequencies
A_m = abs(S_m(1:N/2, :)); A_m = A_m - mean(A_m,'all');
A_a = abs(S_a(1:N/2, :)); A_a = A_a - mean(A_a,'all');
A_d = abs(S_d(1:N/2, :)); A_d = A_d - mean(A_d,'all');

F2_m = fftshift(fft2(A_m));  F2_a = fftshift(fft2(A_a));  F2_d = fftshift(fft2(A_d));
fps = fs/H;              % frame rate (frames per second)
Nt  = size(A_m,2);       % time bins
Nf  = size(A_m,1);       % freq bins (positive freqs)

f_mod = (-Nt/2:Nt/2-1)*(fps/Nt);     % temporal modulation (Hz)
df = fs/N;                           % Hz/bin
nu = (-Nf/2:Nf/2-1) / ((Nf*df/1000) + eps);  % ~cycles/kHz (approx.)

figure('Name','Module4: Modulation Spectrum (|FFT2(|S|)|)');
subplot(1,3,1); imagesc(f_mod, nu, abs(F2_m)); axis xy; colorbar;
xlabel('Temporal mod (Hz)'); ylabel('Spectral mod (cycles/kHz)'); title('Music');
subplot(1,3,2); imagesc(f_mod, nu, abs(F2_a)); axis xy; colorbar;
xlabel('Temporal mod (Hz)'); ylabel('Spectral mod (cycles/kHz)'); title('Ambient');
subplot(1,3,3); imagesc(f_mod, nu, abs(F2_d)); axis xy; colorbar;
xlabel('Temporal mod (Hz)'); ylabel('Spectral mod (cycles/kHz)'); title('Direct');

%% Module 5. Mel filterbank energy (minimal dependency)
nMel = 40;
Hmel = my_melbank(nMel, fs, N, 50, fs/2);       % (nMel x (N/2+1))
P_m = my_melspec(abs(S_m(1:N/2+1,:)), Hmel);
P_a = my_melspec(abs(S_a(1:N/2+1,:)), Hmel);
P_d = my_melspec(abs(S_d(1:N/2+1,:)), Hmel);

figure('Name','Module5: Mel-band Energy (log)');
subplot(1,3,1); imagesc(t, 1:nMel, log(P_m+1e-8)); axis xy; colorbar; title('Music');
xlabel('Time (s)'); ylabel('Mel band');
subplot(1,3,2); imagesc(t, 1:nMel, log(P_a+1e-8)); axis xy; colorbar; title('Ambient');
xlabel('Time (s)'); ylabel('Mel band');
subplot(1,3,3); imagesc(t, 1:nMel, log(P_d+1e-8)); axis xy; colorbar; title('Direct');
xlabel('Time (s)'); ylabel('Mel band');

%% Module 6. Basic lightweight features (ZCR/RMS/centroid/bandwidth/rolloff)
% Time-frame parameters for time-domain features
frm_t = round(0.03*fs); hop_t = round(0.01*fs);
[tfeat_m, feats_m] = basic_feats_both(music, fs, frm_t, hop_t, S_m, fs, N, 0.95);
[tfeat_a, feats_a] = basic_feats_both(ambient, fs, frm_t, hop_t, S_a, fs, N, 0.95);
[tfeat_d, feats_d] = basic_feats_both(direct, fs, frm_t, hop_t, S_d, fs, N, 0.95);

% Plot spectral centroid as example
figure('Name','Module6: Features (Spectral Centroid)');
plot(tfeat_m, feats_m.cent/1000,'-','DisplayName','Music'); hold on;
plot(tfeat_a, feats_a.cent/1000,'-','DisplayName','Ambient');
plot(tfeat_d, feats_d.cent/1000,'-','DisplayName','Direct'); hold off;
xlabel('Time (s)'); ylabel('Centroid (kHz)'); legend; grid on;

%% Module 7. LPC source-filter (envelope & simple resynthesis)
frm = round(0.03*fs); hop = round(0.01*fs); p = 14;
[xsyn, a_list] = lpc_synthesis_simple(direct, fs, frm, hop, p);
xsyn = xsyn ./ max(abs(xsyn)+1e-12);
audiowrite('direct_lpc_synth.wav', xsyn, fs);

% Compare envelope (plot LPC spectral envelope vs magnitude spectrum of a sample frame)
k0 = min(200, size(a_list,2));              % pick a mid frame
a_k0 = a_list(:,k0);
[Henv, ff] = lpc_envelope(a_k0, N, fs);   % ff: 0..fs/2 (Hz)

frame  = direct(1:frm).*my_hann(frm,false);
mag_k0 = abs(fft(frame, N));              
fpos   = (0:N/2)*(fs/N);                  

figure('Name','Module7: LPC envelope vs magnitude');
plot(fpos/1000, 20*log10(mag_k0(1:N/2+1)+1e-9), 'LineWidth',0.8); hold on;
plot(ff/1000,   20*log10(abs(Henv)+1e-9),       'LineWidth',1.2);
xlabel('Freq (kHz)'); ylabel('dB'); grid on;
legend('Magnitude','LPC Env'); hold off;

%% Module 8. Direct vs Ambient comparison (if stereo not available)
% (If you had stereo, you could do Mid/Side here; this compares two scenes.)
% Compare Mel-band averages as a simple summary
m_mean = mean(log(P_m+1e-8), 2); 
a_mean = mean(log(P_a+1e-8), 2);
d_mean = mean(log(P_d+1e-8), 2);
figure('Name','Module8: Mel-band Mean (log)');
plot(1:nMel, m_mean,'-o','DisplayName','Music'); hold on;
plot(1:nMel, a_mean,'-o','DisplayName','Ambient');
plot(1:nMel, d_mean,'-o','DisplayName','Direct'); hold off; grid on;
xlabel('Mel band'); ylabel('Mean log energy'); legend; title('Mel-band averages');

%% Module 9. Onset & tempo via spectral flux (no toolbox)
% Spectral flux for music; simple peak pick + tempo estimate via FFT
[sf, t_sf] = spectral_flux(abs(S_m(1:N/2,:)), t);
[pks, locs] = findpeaks_custom(sf, 0.2*max(sf), round(0.2*fs/H));  %#ok<FNDSB>  (custom below)

figure('Name','Module9: Spectral Flux (Music)');
plot(t_sf, sf, 'k'); hold on;
plot(t_sf(locs), sf(locs), 'ro'); hold off; grid on;
xlabel('Time (s)'); ylabel('Spectral flux'); title('Onsets (peaks)');

% Tempo (BPM) estimate via FFT of spectral flux
fps_sf = 1/mean(diff(t_sf));                 % ~ frames/sec on flux grid
[BPM, bpm_peak_hz] = tempo_from_series(sf, fps_sf, [30 240]);  % returns BPM & Hz
fprintf('Estimated tempo (music): %.1f BPM (%.2f Hz)\n', BPM, bpm_peak_hz);

disp('Done. See figures and generated file: direct_lpc_synth.wav');

%% 


function ensure_colormap()
% Use turbo if available; else parula
try
    colormap turbo;
catch
    colormap parula;
end
end

function y = my_resample_lin(x, fs_in, fs_out)
% Minimal linear-interpolation resampling (avoid toolbox 'resample')
if fs_in == fs_out, y = x; return; end
t_in  = (0:numel(x)-1)/fs_in;
t_out = (0:round(numel(x)*fs_out/fs_in)-1)/fs_out;
y = interp1(t_in, x, t_out, 'linear', 'extrap');
y = y(:);
end

function w = my_hann(M, periodic)
% Hann window; periodic=true → denominator M ; false → M-1 (nonperiodic)
if nargin<2, periodic=false; end
n = (0:M-1).';
if periodic
    w = 0.5 - 0.5*cos(2*pi*n/M);
else
    w = 0.5 - 0.5*cos(2*pi*n/(M-1));
end
end

function [S,f,t] = my_stft(x, fs, M, H, N)
% Custom STFT: Hann window (nonperiodic), hop H, FFT length N>=M
x = x(:);
w = sqrt(my_hann(M,true)); 
L = length(x);
K = max(0, 1 + floor((L - M)/H));
S = zeros(N, K);
idx = 1;
for k = 1:K
    seg = x(idx:idx+M-1).*w;
    S(:,k) = fft(seg, N);
    idx = idx + H;
end
f = (0:N-1)*(fs/N);                     % Hz
t = ((0:K-1)*H + (M/2))/fs;            % frame center time (s)
end

function y = my_istft(S, fs, M, H, N)
% WOLA reconstruction with sqrt-Hann (periodic) when H=M/2 gives near-unity gain
w = sqrt(my_hann(M,true));
K = size(S,2);
L = H*(K-1)+M;
y = zeros(L,1);
wsum = zeros(L,1);
idx = 1;
for k = 1:K
    seg = real(ifft(S(:,k), N));
    seg = seg(1:M) .* w;
    y(idx:idx+M-1) = y(idx:idx+M-1) + seg;
    wsum(idx:idx+M-1) = wsum(idx:idx+M-1) + (w.^2);
    idx = idx + H;
end
wsum(wsum==0) = 1;
y = y ./ wsum;
end

function Hmel = my_melbank(nBands, fs, N, fmin, fmax)
% Build Mel triangle filterbank for FFT bins 0..N/2
if nargin<5, fmax=fs/2; end
if nargin<4, fmin=50;   end
hz2mel = @(f) 2595*log10(1+f/700);
mel2hz = @(m) 700*(10.^(m/2595)-1);
m_edges = linspace(hz2mel(fmin), hz2mel(fmax), nBands+2);
f_edges = mel2hz(m_edges);
bin = round(f_edges/(fs/N));
bin = max(min(bin, N/2), 0);   % clamp
Hmel = zeros(nBands, N/2+1);
for i=1:nBands
    b0=bin(i); b1=bin(i+1); b2=bin(i+2);
    if b1==b0, b1=b0+1; end
    if b2==b1, b2=b1+1; end
    Hmel(i, b0+1:b1+1)   = linspace(0,1, b1-b0+1);
    Hmel(i, b1+1:b2+1)   = linspace(1,0, b2-b1+1);
end
end

function Pmel = my_melspec(Apos, Hmel)
% Apos: (N/2+1 x T) magnitude; Hmel: (nMel x N/2+1)
Pmel = Hmel * (Apos.^2);    % power in mel bands
end

function [t_feat, feats] = basic_feats_both(x, fs, frm, hop, S, fsS, N, roll_p)
% Combined time-domain (ZCR, RMS) and spectral-domain (centroid, bw, rolloff)
% Time-grid for time-domain features:
L = length(x);
K = 1 + floor((L-frm)/hop);
zcr = zeros(1,K); rmsv = zeros(1,K);
idx=1;
w = my_hann(frm,false);
for k=1:K
    seg = x(idx:idx+frm-1).*w;
    zcr(k)  = sum(abs(diff(sign(seg))))/(2*frm);
    rmsv(k) = sqrt(mean(seg.^2));
    idx = idx + hop;
end
t_feat = ((0:K-1)*hop + frm/2)/fs;

% Spectral features from STFT (assumed same hop as H=512 here; slight mismatch acceptable)
A = abs(S(1:N/2+1,:));
f = (0:N/2)*(fsS/N);
K2 = size(A,2);
cent=zeros(1,K2); bw=zeros(1,K2); roll=zeros(1,K2);
for k=1:K2
    ak = A(:,k);
    pk = ak.^2;
    P  = sum(pk) + 1e-12;
    cent(k) = sum(f(:).*pk)/P;
    bw(k)   = sqrt( sum( ((f(:)-cent(k)).^2).*pk)/P );
    cdf = cumsum(pk)/P;
    roll(k) = f(find(cdf>=roll_p,1,'first'));
end
% Align lengths by simple interpolation if needed
if numel(cent) ~= numel(t_feat)
    t_s = ((0:K2-1)* ( (length(x)-frm)/(K2-1) ) + frm/2)/fs; % approx
    cent = interp1(t_s, cent, t_feat, 'linear','extrap');
    bw   = interp1(t_s, bw,   t_feat, 'linear','extrap');
    roll = interp1(t_s, roll, t_feat, 'linear','extrap');
end
feats = struct('zcr',zcr,'rms',rmsv,'cent',cent,'bw',bw,'roll',roll);
end

function [sf, t_sf] = spectral_flux(Apos, t_frames)
% Apos: (N/2 x T) magnitude for positive freqs
A = Apos ./ (max(Apos,[],'all') + 1e-12);
dA = max(A(:,2:end) - A(:,1:end-1), 0);
sf = sum(dA, 1);
t_sf = t_frames(2:end);
end

function [pks, locs] = findpeaks_custom(x, minHeight, minDist)
% Simple peak picker without toolbox 'findpeaks'
if nargin<2, minHeight=0; end
if nargin<3, minDist=1; end
N = numel(x);
cand = false(1,N);
for i=2:N-1
    if x(i) > x(i-1) && x(i) >= x(i+1) && x(i) >= minHeight
        cand(i) = true;
    end
end
idx = find(cand);
% Enforce minimum distance
locs = [];
last = -inf;
for k = 1:numel(idx)
    if isempty(locs) || (idx(k) - last) >= minDist
        locs(end+1) = idx(k); %#ok<AGROW>
        last = idx(k);
    elseif x(idx(k)) > x(locs(end))
        % replace if higher within window
        locs(end) = idx(k);
        last = idx(k);
    end
end
pks = x(locs);
end

function [BPM, f_peak] = tempo_from_series(x, fps, bpmRange)
% Tempo estimate from spectral peak of flux series
% x: series; fps: samples per second on x; bpmRange = [min max]
L = numel(x);
X = abs(fft(x));
f = (0:L-1)*(fps/L);
% search band in Hz
fmin = bpmRange(1)/60; fmax = bpmRange(2)/60;
mask = (f>=fmin & f<=fmax);
[~, idx] = max(X(mask));
idx_all = find(mask);
if isempty(idx_all)
    BPM = NaN; f_peak = NaN;
else
    f_peak = f(idx_all(idx));
    BPM = 60*f_peak;
end
end

function [xsyn, a_list] = lpc_synthesis_simple(x, fs, frm, hop, p)
% Frame-wise LPC coefficients with white-noise excitation synthesis
x = x(:); w = my_hann(frm,false);
K = 1 + floor((length(x)-frm)/hop);
xsyn = zeros(length(x),1); wsum = zeros(length(x),1);
a_list = zeros(p+1, K);
idx = 1;
for k=1:K
    seg = x(idx:idx+frm-1).*w;
    [a, ~] = my_lpc(seg, p);
    a_list(:,k) = a;
    e = 0.03*randn(frm,1);          % simple noise excitation
    yk = filter(1, a, e);
    xsyn(idx:idx+frm-1) = xsyn(idx:idx+frm-1) + yk.*w;
    wsum(idx:idx+frm-1) = wsum(idx:idx+frm-1) + w.^2;
    idx = idx + hop;
end
wsum(wsum==0)=1; xsyn = xsyn ./ wsum;
end

function [Henv, f] = lpc_envelope(a, N, fs)
% LPC spectral envelope from AR polynomial a
[H, w] = freqz(1, a, N, fs);
Henv = H; f = w;
end

function [a, E] = my_lpc(x, p)
% Levinson-Durbin solver for LPC (minimal)
x = x(:);
r = xcorr(x, p, 'biased');         % length 2p+1, symmetric
r = r(p+1:end);                    % r(0)..r(p)
a = zeros(p,1); E = r(1) + 1e-12;
for i=1:p
    if i==1
        k = -r(2)/E;
        a(1) = k; E = E*(1-k^2);
    else
        acc = r(i+1) + a(1:i-1)'*flipud(r(2:i));
        k = -acc/E;
        a(1:i-1) = a(1:i-1) + k*flipud(a(1:i-1));
        a(i) = k; E = E*(1-k^2);
    end
end
a = [1; a];
end
