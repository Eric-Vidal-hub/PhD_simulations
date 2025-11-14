close all;clear all;clc;
function Bvec = Brond(x,y,z)
  r = sqrt(x.^2+y.^2+z.^2);
  Bxx = 0.5*log((r-z)./(r+z));
  Byy = -atan(y.*r./(x.*z));
  Bzz = 0.5*log((r-x)./(r+x));
  len = prod(size(r));
  Bvec = reshape([reshape(Bxx,[len,1]),reshape(Byy,[len,1]),reshape(Bzz,[len,1])],[size(r),3]);
end
function Bvec = Bmagnet(x,y,z,a,b,c,Br)
  Bvec = Br/(4*pi)*(Brond(x-a,y-b,z-c)-Brond(x+a,y-b,z-c)+...
                  Brond(x+a,y+b,z-c)-Brond(x-a,y+b,z-c)+...
                  Brond(x-a,y+b,z+c)-Brond(x-a,y-b,z+c)+...
                  Brond(x+a,y-b,z+c)-Brond(x+a,y+b,z+c));
end
function Bvec = Bdisrot(x,y,z,a,b,c,Br,r0,z0,alpha,beta)
  %% Translation
  xp = x-r0*sin(beta);
  yp = y-r0*cos(beta);
  zp = z-z0;
  %% First rotation 2*beta around z
  xs = cos(2*beta)*xp-sin(2*beta)*yp;
  ys = sin(2*beta)*xp+cos(2*beta)*yp;
  zs = zp;
  %% Second rotation alpha around v = cos(beta) exs + sin(beta) eys
  cb = cos(beta);ca = cos(alpha);sb = sin(beta);sa = sin(alpha);
  xt = (cb^2*(1-ca)+ca)*xs + cb*sb*(1-ca)*ys + sb*sa*zs;
  yt = cb*sb*(1-ca)*xs + (sb^2*(1-ca)+ca)*ys - cb*sa*zs;
  zt = -sb*sa*xs + cb*sa*ys + ca*zs;
  %%
  Bvec = Bmagnet(xt,yt,zt,a,b,c,Br);
  dim = size(Bvec);
  len = prod(dim(1:end-1));
  Bvec = reshape(Bvec,len,3);
  Bxt = Bvec(:,1);
  Byt = Bvec(:,2);
  Bzt = Bvec(:,3);
  %% Undo rotations
  Bxs = (cb^2*(1-ca)+ca)*Bxt + cb*sb*(1-ca)*Byt - sb*sa*Bzt;
  Bys = cb*sb*(1-ca)*Bxt + (sb^2*(1-ca)+ca)*Byt + cb*sa*Bzt;
  Bzs = sb*sa*Bxt - cb*sa*Byt + ca*Bzt;
  %%
  Bx = cos(2*beta)*Bxs+sin(2*beta)*Bys;
  By = -sin(2*beta)*Bxs+cos(2*beta)*Bys;
  Bz = Bzs;
  %%
  Bvec = reshape([Bx,By,Bz],dim);
end
function Bvec = BHallbach8(x,y,z,a,b,c,Br,R0,z0,alpha)
  Bvec = 0;
  for beta=[0,pi/4,pi/2,3*pi/4,pi,5*pi/4,3*pi/2,7*pi/4]
    Btmp = Bdisrot(x,y,z,a,b,c,Br,R0,z0,alpha,beta);
    Bvec = Bvec + Btmp;
  end
end
function res = By(x,y,z,a,b,c,Br,R0,z0,alpha)
  Bvec = BHallbach8(x,y,z,a,b,c,Br,R0,z0,alpha);
  dim = size(Bvec);
  len = prod(dim(1:end-1));
  Bvec = reshape(Bvec,[len,3]);
  res = reshape(Bvec(:,2),dim(1:end-1));
end
%%
% Magnets
a = 6/2;        % mm
b = 6/2;        % mm
c = 128/2;       % mm
Br = 1.08e4;  % G
%%
R0 = 54/2;      % mm
z0 = 0;
alpha = -0.97*pi/180;
%%
%{
xgrid = linspace(-60,60,101);
ygrid = linspace(-60,60,101)';
Bvec = BHallbach8(xgrid,ygrid,0,a,b,c,Br,R0,z0,alpha);
figure(1);
for n=1:3
  subplot(2,2,n);
  imagesc(xgrid,ygrid,Bvec(:,:,n));
  set(gca,'YDir','normal');
end
subplot(2,2,4);
imagesc(xgrid,ygrid,Bvec(:,:,2));%sqrt(sum(Bvec.^2,3)));
set(gca,'YDir','normal','Clim',[0,1300]);
colorbar;
%%
zgrid = linspace(-200,200,1001)';
Bvec = BHallbach8(0,0,zgrid,a,b,c,Br,R0,z0,alpha);
figure(2);clf;
plot(zgrid,squeeze(Bvec(:,1)),zgrid,squeeze(Bvec(:,2)),zgrid,squeeze(Bvec(:,3)),...
      -c*[1,1],[-200,800],'k--',c*[1,1],[-200,800],'k--');
legend('Bx','By','Bz');
%%
ygrid = linspace(-200,200,101);
zgrid = linspace(-200,200,101)';
Bvec = BHallbach8(0,ygrid,zgrid,a,b,c,Br,R0,z0,alpha);
Bvec = squeeze(Bvec);
figure(3);
subplot(2,2,1);
imagesc(xgrid,zgrid,Bvec(:,:,1));
subplot(2,2,2);
imagesc(xgrid,zgrid,Bvec(:,:,2));
subplot(2,2,3);
imagesc(xgrid,zgrid,Bvec(:,:,3));
subplot(2,2,4);
imagesc(xgrid,zgrid,sqrt(sum(Bvec.^2,3)));
%}
zgrid = linspace(-200,200,1001)';
Bvec = BHallbach8(0,0,zgrid,a,b,c,Br,R0,z0,alpha);
figure(1);clf;
plot(zgrid,squeeze(Bvec(:,1)),zgrid,squeeze(Bvec(:,2)),zgrid,squeeze(Bvec(:,3)),...
      -c*[1,1],[-200,800],'k--',c*[1,1],[-200,800],'k--');
legend('Bx','By','Bz');
%%
%
Gamma = 1/4.6e-9;   % Linewidth 1/s
kL = 2*pi/423e-9;   % wavevector m
M = 40*1.66e-27;    % Calcium mass kg
h = 6.626e-34;
hbar = h/(2*pi);
muB = h*1.4e6;      % J/G
kB = 1.38e-23;
gJp = 1;
%%
vL = 50;
v0 = 1000;
%%
Delta_B = hbar*kL*(v0-vL)/(gJp*muB);
B0 = 250;
BL = B0+Delta_B;
%%
eta = 0.75;
L = M*v0^2/(hbar*kL*Gamma*eta);
%%
delta_L=-kL*(v0+vL)/2-gJp*muB/hbar*(BL+B0)/2;
disp(['Velocity from ',num2str(v0),' to ',num2str(vL),' m/s']);
disp(['Bfield from ',num2str(B0),' to ',num2str(BL),' G']);
disp(['detuning ',num2str(delta_L/(2*pi*1e9)),' GHz']);
disp(['length ',num2str(L*100),' cm (eta=',num2str(eta),')']);
Bid = @(z,L) (BL+(B0-BL)*sqrt(1-z/L)).*(z>=0).*(z<=L);
%%
R0s = linspace(19/2+2+a,50,101)';
Bvec = BHallbach8(0,0,c/2,a,b,c,Br,R0s,0,0);
figure(2);clf;
plot(R0s,squeeze(Bvec(:,2)),[0,50],[BL,BL],'r--');
%%
zgrid = L*linspace(-0.3,1.3,1001);
% Magnets
a = 6e-3/2;        % mm
b = 6e-3/2;        % mm
Br = 1.08e4;  % G
%%
% By(x,y,z,a,b,c,Br,R0,z0,alpha)
%%
f = @(p) Bid(zgrid,L*1.05) - By(0,0,zgrid,a,b,p(1),Br,p(2),p(3),p(4));
par0 = [L/2;0.028;L/2;-5*pi/180];
parfit = lsqnonlin(f,par0);
figure(3);clf;
subplot(2,1,1);
plot(zgrid,Bid(zgrid,L),'r--',zgrid,By(0,0,zgrid,a,b,parfit(1),Br,parfit(2),parfit(3),parfit(4)),'b-');
subplot(2,1,2);
plot(parfit(3)+parfit(1)*[-1,1],parfit(2)+sin(parfit(4))*parfit(1)*[-1,1],'r--');%,...
      %parfit(3)+parfit(1)+parfit(7)+parfit(7)*[-1,1],parfit(5)+sin(parfit(6))*parfit(7)*[-1,1],'r--');
set(gca,'XLim',[-0.1,0.4]);
%%
R2 = 27e-3;
c2 = 6e-3;
z2 = L;
alpha2 = -5*pi/180;
figure(4);clf;
subplot(2,1,1);
plot(zgrid,Bid(zgrid,L),'r--',...
    zgrid,By(0,0,zgrid,a,b,parfit(1),Br,parfit(2),parfit(3),parfit(4)),'b-',...
    zgrid,By(0,0,zgrid,a,b,parfit(1),Br,parfit(2),parfit(3),parfit(4))+By(0,0,zgrid,a,b,c2,Br,R2,z2,alpha2),'m-');
subplot(2,1,2);
plot([-0.1,0.4],1e-3*(19/2+2)*[1,1],'k--',...
      parfit(3)+parfit(1)*[-1,1],parfit(2)+sin(parfit(4))*parfit(1)*[-1,1],'r--');
hold on;
fill(parfit(3)+parfit(1)*[-1,1,1,-1]+a*sin(parfit(4))*[1,1,-1,-1],...
     parfit(2)+sin(parfit(4))*parfit(1)*[-1,1,1,-1]+a*cos(parfit(4))*[-1,-1,1,1],'r','FaceAlpha',0.2);
fill(z2+c2*[-1,1,1,-1],R2+a*[-1,-1,1,1],'m','FaceAlpha',0.2);
hold off;

