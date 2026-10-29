% function pdex1_sproul
clc
clear all
% close all
%loadexperimentaldata_0201_24
global Sl Sr Dn Dp taub krad ni alpha d nint0 beta_n beta_p N_trap e_n e_p N_A N_D
global beta_n2 beta_p2 N_trap2 e_n2 e_p2 beta_n3 beta_p3 N_trap3 e_n3 e_p3 n0 p0 ntrap0_1 ntrap0_2 ntrap0_3
% data_trPL=load('taudiff.dat');
% data_Ratio=load('RATIO.dat');

m = 0;  % means slab and no cylindrical coordinates or the like
q=1.6e-19;
kT=0.0258; %kT or kT/q (depending on context) in eV or V
Eg=1.625;% band gap in eV %%YY
Nc=2.2e18;
Nv=2.2e18;
ni=sqrt(Nc*Nv*exp(-Eg/kT));
N_A = 0; % p doping density in cm^-3, not implemented yet
N_D = 0; % n doping density in cm^-3, not implemented yet


taub=inf;%1e-3; %bulk  lifetime in seconds
krad =1e-11;%3e-11%1e-11;%%krad  ！第5轮，调形状
d=450e-7;%450e-7;%650e-7;   %thickness in cm %%YY
Jscfilm = 0.0247; %%%assumption. The film would absorb about 20mA/cm? worth of photons at one sun. This is just a rough estimate for now.
Gonesun = Jscfilm/q/d;

% kdiff_set = 3e-11;
taun_set = 1e-7;
taup_set = 1e-6;
taun_set_2 = 2e-8;
taun_set_3 = 1e-6;

E_trap=Eg/2;%1.8357%1.53;%1.5790;
n1_1 = Nc*exp((E_trap-Eg)/kT);
p1_1 = Nv*exp((-E_trap)/kT);
N_trap=1e+17;%1.4137e+18;%1.31e18;%1.3025e+18;%cm^-3
if N_trap==0
    beta_n= 0;
    beta_p= 0;
else
    beta_n= 1/taun_set/N_trap; %1.15e-10;%1.6136e-12;%5.03e-10;%5.1463e-10;%cm3/s
    beta_p= 1/taup_set/N_trap; %1.15e-10;%1.6136e-12;%5.03e-10;%5.1463e-10;%cm3/s
    % beta_n= 1.15e-12;%1.6136e-12;%5.03e-10;%5.1463e-10;%cm3/s
    % beta_p= 1.68e-12;%5.5415e-11;%3.7307e-11;%cm3/s
    % beta_p= kdiff_set*n1_1/N_trap; %1.68e-9;%5.5415e-11;%3.7307e-11;%cm3/s
end

E_trap2=Eg/2;%1.4870;
n1_2 = Nc*exp((E_trap2-Eg)/kT);
p1_2 = Nv*exp((-E_trap2)/kT);
N_trap2=0e15;

%1.4e18%1.2015e+18%1.6285e+18 %1.35e18;%1.3944e+18;%cm^-3
% beta_n2=1/taun_set_2/N_trap;%9.26e-12;%1.0266e-12;%1.9882e-11;%cm3/s
% beta_p2= 1/taun_set_2/N_trap; %1.15e-10;%1.6136e-12;%5.03e-10;%5.1463e-10;%cm3/s
% beta_p2=kdiff_set*n1_2/N_trap/10; %5.36e-11;%8.1915e-11;%6.6666e-11;%cm3/s
beta_n2=0;%1.0266e-12;%1.9882e-11;%cm3/s
beta_p2=0;%8.1915e-11;%6.6666e-11;%cm3/s

E_trap3=1.4;%4.8298e-11 ?????????%1.57%1.567;%1.5790;
n1_3 = Nc*exp((E_trap3-Eg)/kT);
% n1_3 = 1e14;
% E_trap3 = Eg+kT*log(n1_3/Nc);
p1_3 = Nv*exp((-E_trap3)/kT);
N_trap3=0e16;%1.58e18%1.1667e+18;%1.31e18;%1.3025e+18;%cm^-3
if N_trap3==0
    beta_n3= 0;
    beta_p3= 0;
else
    beta_n3=1/taun_set_3/N_trap3*1;%5.03e-10;%5.1463e-10;%cm3/s
    beta_p3=1/taun_set_3/N_trap3*1;%3.7307e-11;%cm3/s
end
% beta_n3=9.26e-17;%1.0266e-12;%1.9882e-11;%cm3/s
% beta_p3=5.36e-17;%8.1915e-11;%6.6666e-11;%cm3/s

Sl=0;  %surface recombination velocity on the left ！第4轮调平衡时的Ratio大小时确定。
Sr=0;%YY%surface recombination velocity on the right
alpha=1.3e0;  %absorption coefficient in 1/cm at the wavelength of the excitation, 4.30e5



% N_trap=6.8775e+35/n1_1

tau_SHR_n_1 = 1/N_trap/beta_n;
tau_SHR_p_1 = 1/N_trap/beta_p;
tau_SHR_n_2 = 1/N_trap2/beta_n2;
tau_SHR_p_2 = 1/N_trap2/beta_p2;
tau_SHR_n_3 = 1/N_trap3/beta_n3;
tau_SHR_p_3 = 1/N_trap3/beta_p3;


ktot = krad + 1/n1_1/tau_SHR_p_1 + 1/n1_2/tau_SHR_p_2 + 1/n1_3/tau_SHR_p_3;

e_n=beta_n*n1_1;
e_p=beta_p*p1_1;
e_n2=beta_n2*n1_2;
e_p2=beta_p2*p1_2;
e_n3=beta_n3*n1_3;
e_p3=beta_p3*p1_3;

Earr=linspace(1.51, 1.79, 400); %energy axis around the peak
Phibb=Earr.^2.*exp(-Earr/kT); %only proportional to the BB spectrum. Energy independent prefactors were ignored.
alpha_exp=load('alpha_vs_E_SSPL.dat'); %%old data on MAPI from 2015 or so.
alphaE=interp1(alpha_exp(:,1), alpha_exp(:,2), Earr);%interpolate experimental data onto new energy axis
%  t = logspace(-9,-6,100);
t = logspace(-11,-2,512);
% t=[1e-10,1e-9, 2e-9, 4e-9, 10e-9]%%人为选择时间
%%%%%%%%%%%%%%%
x = linspace(0,d,150);


% [n0t,p0t, ntrap_acc1t, ntrap_acc2t, ntrap_acc3t,ntrap_acc4t] = calcequil_3traps(n0,Eg, Nv, N_trap, N_trap2, E_trap, E_trap2, N_trap3, E_trap3, 0, 0)

[n0,p0, ntrap0_1, ntrap0_2, ntrap0_3] = calcequil_3traps(ni,Eg, Nv, N_trap, N_trap2, E_trap, E_trap2, N_trap3, E_trap3); 
% ntrap0_3_anal = sqrt(Nv*N_trap3)*exp(-(E_trap3)/2/kT);       % assuming n0 << p0 = ntrap0_3

% nstartarr= 1e18*10.^-[4,2,1,0];  %%n_pulse cm-3   %carrier concentraation at the beginnning of the pulse in 1/cm?
nstartarr= 1.6e17;  %%n_pulse cm-3   %carrier concentraation at the beginnning of the pulse in 1/cm?
% nstartarr= 1e16*[1,2.15,6.15,17.225];  %%n_pulse cm-3   %carrier concentraation at the beginnning of the pulse in 1/cm?
% arrarr=[0.001,0.01,0.1,1,10,100,1000000];       % array for mobilities
arrarr=[100000];       % array for mobilities

mu_arr    = [1e0];      % mobilities
alpha_arr = [1e4,5e4, 1e5, 5e5, 1e6];   % absorption coefficients
foldername = 'simulation_results-trap-high-Nt_alphas-mu-1';
mkdir(foldername);

for XX = 1:length(mu_arr)
    mu = mu_arr(XX);

    for j = 1:length(alpha_arr)
        alpha = alpha_arr(j);

        fprintf('mu = %g, alpha = %g\n', mu, alpha)

        D= kT*mu; %diffusion coefficient in cm?/s
        Dn=D;
        Dp=D;

        % nstartarr=logspace(15, 17, 1);
        %nstartarr=6.713810390029847E15%5.6153410969043064E16%8.48e16%1.96e16  %1.96e17%1.8e17; %initial n_pluse 匹配ΔEF

        for ii=1:length(nstartarr)
            nint0=nstartarr(ii)%%%carrier concentraation at the beginnning of the pulse in 1/cm?
            %%%%%%%%%%%%%%%%%%%illumination from the left (use pdexlic)
            sol= pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
            % Extract the first solution component as u.  This is not necessary
            % for a single equation, but makes a point about the form of the output.
            n = sol(:,:,1); %electron concentration n(x,t)
            p = sol(:,:,2); %hole concentration p(x,t)
            nt1 = sol(:,:,3); %hole concentration p(x,t)
            nt2 = sol(:,:,4); %hole concentration p(x,t)
            nt3 = sol(:,:,5); %hole concentration p(x,t)
            nint(:,ii)=trapz(x, n')'/d; %space-integrated electron concentration n(t)
            pint(:,ii)=trapz(x, p')'/d;
            nt1_int(:,ii)=trapz(x, nt1')'/d; %space-integrated occupied trap concentration nt(t)
            nt2_int(:,ii)=trapz(x, nt2')'/d; %space-integrated occupied trap concentration nt(t)
            nt3_int(:,ii)=trapz(x, nt3')'/d; %space-integrated occupied trap concentration nt(t)
    %         PL(:,ii)=nint(:,ii).*pint(:,ii)-ni^2; % wrong wrong
            PL_wo_reabs(:,ii)=trapz(x,n'.*p')'/d-ni^2;   % include diffusion, ignores reabsorb

            %%%%%%%%%%%%%% calculate spectral shifts
            %         disp('calculate spectral shifts left')
            for kk=1:length(t)
                for zz=1:length(Earr)
                    PL_spec(zz,kk)=trapz(x, alphaE(zz).*Phibb(zz).*exp(-alphaE(zz)*x).*n(kk,:).*p(kk,:))/d;
                end
                %first three factors are emssion weighting. exp(-alpha...)
                %escape probability (e-h must go through film first to create
                %pl, alpha_e is the absorption coefficient and due to
                %reciprocity principle it is also the emission probability and
                %Phibb is the black body spectron andhoton emission rate 
                %depends on the density of optical states and occupation
                PL(kk,ii) = trapz(Earr,PL_spec(:,kk));
                PL_spec_norm(:,kk)=PL_spec(:,kk)./max(PL_spec(:,kk));
                %%YYadd to cal ratio%%%
                PL_spec_norm_with_Earr(:,1)=Earr;
                PL_spec_norm_with_Earr(:,2)=PL_spec_norm(:,kk);
                E_range_A = find(PL_spec_norm_with_Earr(:,1) >= 1.6 & PL_spec_norm_with_Earr(:,1) <= 1.63);
                E_range_B = find(PL_spec_norm_with_Earr(:,1) >= 1.7 & PL_spec_norm_with_Earr(:,1) <= 1.73);
                RATIO(:,1) = t*1e9;
                RATIO(kk,ii,XX+1) = trapz(PL_spec_norm_with_Earr(E_range_A,1), PL_spec_norm_with_Earr(E_range_A,2))./trapz(PL_spec_norm_with_Earr(E_range_B,1), PL_spec_norm_with_Earr(E_range_B,2));

                %%YYadd end%%

            end
            PL(:,ii) = PL(:,ii) /max(PL(:,ii)) *nint0^2 - ni^2;
            Vint(:,ii)=kT*log(PL(:,ii)/ni^2+1);%internal voltage assuming high level injection
            tau(:,ii)=-2*gradient(t, log(PL(:,ii)));
            tau_n(:,ii) = -nint(:,ii)./gradient(nint(:,ii),t);
            tau_p(:,ii) = -pint(:,ii)./gradient(pint(:,ii),t);
            k_diff(:,ii)= -1e16 ./ ( 2*(PL(:,ii)).^0.5 ) .* gradient(log(PL(:,ii)),t)/1e16;  % multiplication and division with 1e16 to avoid numerical errors
            tauS=2*d/Sr+4*d^2/D/pi^2;%%%Sproul equation for asymmetric S
            %         tauanal(:,ii)=(k*nint(:,ii)+1/taub+1/tauS).^(-1); %analytical approximation
            tau_anal_SRH_1 = tau_SHR_n_1+tau_SHR_p_1 + tau_SHR_n_1*p1_1/nint(:,ii) + tau_SHR_p_1*n1_1/nint(:,ii);
            tau_anal_SRH_2 = tau_SHR_n_2+tau_SHR_p_2 + tau_SHR_n_2*p1_2/nint(:,ii) + tau_SHR_p_2*n1_2/nint(:,ii);
            tau_anal_SRH_3 = tau_SHR_n_3+tau_SHR_p_3 + tau_SHR_n_3*p1_3/nint(:,ii) + tau_SHR_p_3*n1_3/nint(:,ii);
            tauanal(:,ii)=(krad*nint(:,ii)'+1/taub+1/tauS+1./tau_anal_SRH_1+1./tau_anal_SRH_2+1./tau_anal_SRH_3).^(-1); %analytical approximation
            %         %%%%%%%%%%%%%%%%%%%illumination from the right (use pdexlic2, rest the
            %         %%%%%%%%%%%%%%%%%%%same as above)
            %         sol2 = pdepe(m,@pdex1pde,@pdex1ic2,@pdex1bc,x,t);
            %         % Extract the first solution component as u.  This is not necessary
            %         % for a single equation, but makes a point about the form of the output.
            %         u2 = sol2(:,:,1);%%n(x)
            %         up2 = sol2(:,:,2);%%p(x)
            %         nint2(:,ii)=trapz(x, u2')'/d;
            %         pint2(:,ii)=trapz(x, up2')'/d;
            %         PL2(:,ii)=nint2(:,ii).*pint2(:,ii)-ni^2;
            %         Vint2(:,ii)=kT*log(PL2(:,ii)/ni^2+1);%internal voltage assuming high level injection
            %         tau2(:,ii)=-2*gradient(t, log(PL2(:,ii)));
            %         PL2normalized(:,ii)= PL2(:,ii)/PL2(1,ii);%%YY add only try
            %         tauS2=2*d/Sl+4*d^2/D/pi^2;%%%Sproul equation for asymmetric S
            %         tauanal2(:,ii)=(k*nint2(:,ii)+1/taub+1/tauS2).^(-1); %analytical approximation
            %
            %         %%%%%%%%%%%%%% calculate spectral shifts
            %         %disp('calculate spectral shifts right')
            %         for kk=1:length(t)
            %             for zz=1:length(Earr)
            %                 PL_spec2(zz,kk)=trapz(x, alphaE(zz).*Phibb(zz).*exp(-alphaE(zz)*(d-x)).*u2(kk,:).*up2(kk,:));
            %             end
            %             PL_spec_norm2(:,kk)=PL_spec2(:,kk)./max(PL_spec2(:,kk));
            %             %%YYadd to cal ratio%%%
            %             PL_spec_norm_with_Earr2(:,1)=Earr;
            %             PL_spec_norm_with_Earr2(:,2)=PL_spec_norm2(:,kk);
            %             E_range_A2 = find(PL_spec_norm_with_Earr2(:,1) >= 1.6 & PL_spec_norm_with_Earr2(:,1) <= 1.63);
            %             E_range_B2 = find(PL_spec_norm_with_Earr2(:,1) >= 1.7 & PL_spec_norm_with_Earr2(:,1) <= 1.73);
            %             RATIO2(:,1) = t*1e9;
            %             RATIO2(kk,XX+1) = trapz(PL_spec_norm_with_Earr2(E_range_A,1), PL_spec_norm_with_Earr2(E_range_A,2))./trapz(PL_spec_norm_with_Earr2(E_range_B,1), PL_spec_norm_with_Earr2(E_range_B,2));
            %
            %             %%YYadd end%%
            %
            %         end
            %
            %         %%%%ratio illumination from left to illumination from right
            %         ratPL(:,ii)=PL(:,ii)./PL2(:,ii);
            %         avratPL(ii)=sum(abs(ratPL(:,ii)-1));
            %
            %         % A surface plot is often a good way to study a solution.
            %         figure(1)
            %         surf(x,log10(t),(u));
            %         % title('Numerical solution computed with 20 mesh points.');
            %         xlabel('Distance x');
            %         ylabel('Time t');

            %         figure(11)
            %         %         subplot(1,2,1)
            %         plot(x, n, x, p, '.')
            %         %         hold on
            %         xlabel('distance x (cm)')
            %         ylabel('carrier density n (1/cm?)')
            %         subplot(1,2,2)
            %         plot(x, u2, x, up2, '.')
            %         xlabel('distance x (cm)')
            %         ylabel('carrier density n (1/cm?)')

            %         figure(33)
            %         plot(Earr, PL_spec_norm)
            %         xlabel('energy E (eV)')
            %         ylabel('PL (norm)')
            %
            %         figure(34)
            %         imagesc(Earr, t*1e6, PL_spec')
            %         colorbar
            %         xlabel('energy E (eV)')
            %         ylabel('time t ( )')

        end
        
        filename = sprintf('alpha-%g-mu-%g.csv', alpha, mu);
        filepath = fullfile(foldername, filename);
        T = table(t(:), PL(:)/max(PL), Vint(:), tau(:), ...
        'VariableNames', {'time_s', 'PL_norm', 'QFLS_eV', 'tau_diff_s'});
        writetable(T, filepath);
    end
end

for XX=1:length(arrarr)
    mu=arrarr(XX)
    
    %%%%%%%%%%
    %%%YYend%%
    
    D= kT*mu; %diffusion coefficient in cm?/s
    Dn=D;
    Dp=D;
    
    % nstartarr=logspace(15, 17, 1);
    %nstartarr=6.713810390029847E15%5.6153410969043064E16%8.48e16%1.96e16  %1.96e17%1.8e17; %initial n_pluse 匹配ΔEF
    
    for ii=1:length(nstartarr)
        nint0=nstartarr(ii)%%%carrier concentraation at the beginnning of the pulse in 1/cm?
        %%%%%%%%%%%%%%%%%%%illumination from the left (use pdexlic)
        sol= pdepe(m,@pdex1pde,@pdex1ic,@pdex1bc,x,t);
        % Extract the first solution component as u.  This is not necessary
        % for a single equation, but makes a point about the form of the output.
        n = sol(:,:,1); %electron concentration n(x,t)
        p = sol(:,:,2); %hole concentration p(x,t)
        nt1 = sol(:,:,3); %hole concentration p(x,t)
        nt2 = sol(:,:,4); %hole concentration p(x,t)
        nt3 = sol(:,:,5); %hole concentration p(x,t)
        nint(:,ii)=trapz(x, n')'/d; %space-integrated electron concentration n(t)
        pint(:,ii)=trapz(x, p')'/d;
        nt1_int(:,ii)=trapz(x, nt1')'/d; %space-integrated occupied trap concentration nt(t)
        nt2_int(:,ii)=trapz(x, nt2')'/d; %space-integrated occupied trap concentration nt(t)
        nt3_int(:,ii)=trapz(x, nt3')'/d; %space-integrated occupied trap concentration nt(t)
%         PL(:,ii)=nint(:,ii).*pint(:,ii)-ni^2; % wrong wrong
        PL_wo_reabs(:,ii)=trapz(x,n'.*p')'/d-ni^2;   % include diffusion, ignores reabsorb

        %%%%%%%%%%%%%% calculate spectral shifts
        %         disp('calculate spectral shifts left')
        for kk=1:length(t)
            for zz=1:length(Earr)
                PL_spec(zz,kk)=trapz(x, alphaE(zz).*Phibb(zz).*exp(-alphaE(zz)*x).*n(kk,:).*p(kk,:))/d;
            end
            %first three factors are emssion weighting. exp(-alpha...)
            %escape probability (e-h must go through film first to create
            %pl, alpha_e is the absorption coefficient and due to
            %reciprocity principle it is also the emission probability and
            %Phibb is the black body spectron andhoton emission rate 
            %depends on the density of optical states and occupation
            PL(kk,ii) = trapz(Earr,PL_spec(:,kk));
            PL_spec_norm(:,kk)=PL_spec(:,kk)./max(PL_spec(:,kk));
            %%YYadd to cal ratio%%%
            PL_spec_norm_with_Earr(:,1)=Earr;
            PL_spec_norm_with_Earr(:,2)=PL_spec_norm(:,kk);
            E_range_A = find(PL_spec_norm_with_Earr(:,1) >= 1.6 & PL_spec_norm_with_Earr(:,1) <= 1.63);
            E_range_B = find(PL_spec_norm_with_Earr(:,1) >= 1.7 & PL_spec_norm_with_Earr(:,1) <= 1.73);
            RATIO(:,1) = t*1e9;
            RATIO(kk,ii,XX+1) = trapz(PL_spec_norm_with_Earr(E_range_A,1), PL_spec_norm_with_Earr(E_range_A,2))./trapz(PL_spec_norm_with_Earr(E_range_B,1), PL_spec_norm_with_Earr(E_range_B,2));
            
            %%YYadd end%%
            
        end
        PL(:,ii) = PL(:,ii) /max(PL(:,ii)) *nint0^2 - ni^2;
        Vint(:,ii)=kT*log(PL(:,ii)/ni^2+1);%internal voltage assuming high level injection
        tau(:,ii)=-2*gradient(t, log(PL(:,ii)));
        tau_n(:,ii) = -nint(:,ii)./gradient(nint(:,ii),t);
        tau_p(:,ii) = -pint(:,ii)./gradient(pint(:,ii),t);
        k_diff(:,ii)= -1e16 ./ ( 2*(PL(:,ii)).^0.5 ) .* gradient(log(PL(:,ii)),t)/1e16;  % multiplication and division with 1e16 to avoid numerical errors
        tauS=2*d/Sr+4*d^2/D/pi^2;%%%Sproul equation for asymmetric S
        %         tauanal(:,ii)=(k*nint(:,ii)+1/taub+1/tauS).^(-1); %analytical approximation
        tau_anal_SRH_1 = tau_SHR_n_1+tau_SHR_p_1 + tau_SHR_n_1*p1_1/nint(:,ii) + tau_SHR_p_1*n1_1/nint(:,ii);
        tau_anal_SRH_2 = tau_SHR_n_2+tau_SHR_p_2 + tau_SHR_n_2*p1_2/nint(:,ii) + tau_SHR_p_2*n1_2/nint(:,ii);
        tau_anal_SRH_3 = tau_SHR_n_3+tau_SHR_p_3 + tau_SHR_n_3*p1_3/nint(:,ii) + tau_SHR_p_3*n1_3/nint(:,ii);
        tauanal(:,ii)=(krad*nint(:,ii)'+1/taub+1/tauS+1./tau_anal_SRH_1+1./tau_anal_SRH_2+1./tau_anal_SRH_3).^(-1); %analytical approximation
        %         %%%%%%%%%%%%%%%%%%%illumination from the right (use pdexlic2, rest the
        %         %%%%%%%%%%%%%%%%%%%same as above)
        %         sol2 = pdepe(m,@pdex1pde,@pdex1ic2,@pdex1bc,x,t);
        %         % Extract the first solution component as u.  This is not necessary
        %         % for a single equation, but makes a point about the form of the output.
        %         u2 = sol2(:,:,1);%%n(x)
        %         up2 = sol2(:,:,2);%%p(x)
        %         nint2(:,ii)=trapz(x, u2')'/d;
        %         pint2(:,ii)=trapz(x, up2')'/d;
        %         PL2(:,ii)=nint2(:,ii).*pint2(:,ii)-ni^2;
        %         Vint2(:,ii)=kT*log(PL2(:,ii)/ni^2+1);%internal voltage assuming high level injection
        %         tau2(:,ii)=-2*gradient(t, log(PL2(:,ii)));
        %         PL2normalized(:,ii)= PL2(:,ii)/PL2(1,ii);%%YY add only try
        %         tauS2=2*d/Sl+4*d^2/D/pi^2;%%%Sproul equation for asymmetric S
        %         tauanal2(:,ii)=(k*nint2(:,ii)+1/taub+1/tauS2).^(-1); %analytical approximation
        %
        %         %%%%%%%%%%%%%% calculate spectral shifts
        %         %disp('calculate spectral shifts right')
        %         for kk=1:length(t)
        %             for zz=1:length(Earr)
        %                 PL_spec2(zz,kk)=trapz(x, alphaE(zz).*Phibb(zz).*exp(-alphaE(zz)*(d-x)).*u2(kk,:).*up2(kk,:));
        %             end
        %             PL_spec_norm2(:,kk)=PL_spec2(:,kk)./max(PL_spec2(:,kk));
        %             %%YYadd to cal ratio%%%
        %             PL_spec_norm_with_Earr2(:,1)=Earr;
        %             PL_spec_norm_with_Earr2(:,2)=PL_spec_norm2(:,kk);
        %             E_range_A2 = find(PL_spec_norm_with_Earr2(:,1) >= 1.6 & PL_spec_norm_with_Earr2(:,1) <= 1.63);
        %             E_range_B2 = find(PL_spec_norm_with_Earr2(:,1) >= 1.7 & PL_spec_norm_with_Earr2(:,1) <= 1.73);
        %             RATIO2(:,1) = t*1e9;
        %             RATIO2(kk,XX+1) = trapz(PL_spec_norm_with_Earr2(E_range_A,1), PL_spec_norm_with_Earr2(E_range_A,2))./trapz(PL_spec_norm_with_Earr2(E_range_B,1), PL_spec_norm_with_Earr2(E_range_B,2));
        %
        %             %%YYadd end%%
        %
        %         end
        %
        %         %%%%ratio illumination from left to illumination from right
        %         ratPL(:,ii)=PL(:,ii)./PL2(:,ii);
        %         avratPL(ii)=sum(abs(ratPL(:,ii)-1));
        %
        %         % A surface plot is often a good way to study a solution.
        %         figure(1)
        %         surf(x,log10(t),(u));
        %         % title('Numerical solution computed with 20 mesh points.');
        %         xlabel('Distance x');
        %         ylabel('Time t');
        
        %         figure(11)
        %         %         subplot(1,2,1)
        %         plot(x, n, x, p, '.')
        %         %         hold on
        %         xlabel('distance x (cm)')
        %         ylabel('carrier density n (1/cm?)')
        %         subplot(1,2,2)
        %         plot(x, u2, x, up2, '.')
        %         xlabel('distance x (cm)')
        %         ylabel('carrier density n (1/cm?)')
        
        %         figure(33)
        %         plot(Earr, PL_spec_norm)
        %         xlabel('energy E (eV)')
        %         ylabel('PL (norm)')
        %
        %         figure(34)
        %         imagesc(Earr, t*1e6, PL_spec')
        %         colorbar
        %         xlabel('energy E (eV)')
        %         ylabel('time t ( )')
        
    end
    
    T = table(10.^t(:), PL(:)/max(PL), Vint(:), tau(:), ...
    'VariableNames', {'time_s', 'PL_norm', 'QFLS_eV', 'tau_diff_s'});
    filename = sprintf('alpha-%g-mu-%g.csv', alpha, mu);
    writetable(T, filename);
    % all not position dependent!
    Rrad=krad*nint.*pint; %radiative recombination rate
    Rsrh1=N_trap*beta_n*beta_p*(nint.*pint-ni^2)./(nint*beta_n+pint*beta_p+e_n+e_p); %SRH recombination rate of trap 1
    Rsrh2=N_trap2*beta_n2*beta_p2*(nint.*pint-ni^2)./(nint*beta_n2+pint*beta_p2+e_n2+e_p2); %SRH recombination rate of trap 2
    Rsrh3=N_trap3*beta_n3*beta_p3*(nint.*pint-ni^2)./(nint*beta_n3+pint*beta_p3+e_n3+e_p3);
    Rtot=Rrad+Rsrh1+Rsrh2+Rsrh3; %%%Rtot=Rrad+Rsrh1+Rsrh2
    %%%relating it to illumination
    suns=Rtot/Gonesun;
    %     sunsinter=interp1(Vint, suns, dEfexp); %%interpolated intensity
    PLQY = Rrad./Rtot;
    
    photoconductivity = mu*nint+mu*pint;
    
%     t_trap_equil = tau_SHR_n_1*(log(nstartarr/N_trap)+E_trap/2/kT);
    %% Figures
    
    % PLQY vs intensity (probably still wrong)
%         figure
%         loglog(suns,PLQY)
%         xlabel('suns');
%         ylabel('PLQY');
    %
    
    % photoconductivity vs time
    figure
    loglog(t,photoconductivity)
    xlabel('Time t');
    ylabel('photoconductivity');
    xlim([1e-9 3e-4])

    % charge carrier density vs time
    figure('Units','pixels','Position',[300 100 833 800]);
    sgtitle({['Carrier concentration vs time'],[ '{\fontsize{11}',...
            'N_{trap} = ',sprintf('%.0e',N_trap3),' cm^{-3}, ',...
            'n_{1} = ',sprintf('%.0e',n1_3),' cm^{-3}, ',...
            '\tau_{n} = ',sprintf('%.0e',tau_SHR_n_3),' s, ',...
            '\tau_{p} = ',sprintf('%.0e',tau_SHR_p_3),' s',...
            '}']}) 
    for idx = 1:1
        subplot(2,2,idx)
        %         loglog(t, nint, t, pint,t,(nint.*pint).^0.5)
        %         loglog(t, nint, t, pint)
        %             loglog(t, nint+nt1_int+nt2_int+nt3_int, t, pint,'--')
        %             idx=4;
%         loglog(t, nint(:,idx), t,nt1_int(:,idx), t,nt2_int(:,idx), t,nt3_int(:,idx), t, pint(:,idx),'--')
        loglog(t, nint(:,idx), t,nt3_int(:,idx), t, pint(:,idx),'--')
        hold on
        loglog(t,n1_3*t./t,':')
        loglog(t,N_trap3*t./t,':')
%         loglog(t,n1_3*nt3_int(:,idx)./(N_trap3-nt3_int(:,idx)),'-.')
%         loglog(t,n1_3*nt3_int(:,idx)./(N_trap3),'-.')
%         loglog(t,N_trap3.*nint(:,idx)./(n1_3+nint(:,idx)),'-.')
        loglog(t,N_trap3.*nint(:,idx)./(n1_3),'-.')
%         loglog(t,N_trap3*1./(1+(beta_p3*pint(:,idx))./(beta_n3.*nint(:,idx))),'-.')
        loglog(t,N_trap3*1./(1+(beta_p3)./(beta_n3)).*t./t,'-.')
%         loglog(t,nstartarr(idx)*(1-exp(-t/tau_SHR_n_3))+nt3_int(1,idx),'-.')
%         loglog(t,N_trap3./(1+n1_3./nint(:,idx)+beta_p3*pint(:,idx)/beta_n3./nint(:,idx)),'-.')
%         loglog(t,N_trap3./(1+beta_p3/beta_n3).*t./t,'-.')
%         loglog(t,n1_3*nt3_int(:,idx)./(N_trap3),'-.')
        xlabel('Time t');
        ylabel('charge carrier concentration n (cm^{-3})');
        legend('n','nt3','p','n1_3','Nt_3','Nt*n/n1','Nt/(1+\beta_p/\beta_n)','Location','best')
        title(['n(t=0) = ',num2str(nstartarr(idx),'%.E'),' cm^{-3}'])
            xlim([1e-9 3e-4])
%         ylim([1e12 1e18])
    end

%         % trapping rates vs time
%     figure('Units','pixels','Position',[300 100 833 800]);
%     sgtitle({['Trapping rates vs time'],[ '{\fontsize{11}',...
%             'N_{trap} = ',sprintf('%.0e',N_trap3),' cm^{-3}, ',...
%             'n_{1} = ',sprintf('%.0e',n1_3),' cm^{-3}, ',...
%             '\tau_{n} = ',sprintf('%.0e',tau_SHR_n_3),' s, ',...
%             '\tau_{p} = ',sprintf('%.0e',tau_SHR_p_3),' s',...
%             '}']}) 
%     for idx = 1:4
%         subplot(2,2,idx)
%         %         loglog(t, nint, t, pint,t,(nint.*pint).^0.5)
%         %         loglog(t, nint, t, pint)
%         %             loglog(t, nint+nt1_int+nt2_int+nt3_int, t, pint,'--')
%         %             idx=4;
%         trapping_e = nint(:,idx).* (N_trap3-nt3_int(:,idx))/N_trap3/tau_SHR_n_3;
%         detrapping_e = n1_3*nt3_int(:,idx)/N_trap3/tau_SHR_n_3;
%         trapping_p = pint(:,idx).* nt3_int(:,idx)/N_trap3/tau_SHR_p_3;
%         detrapping_p = p1_3*(N_trap3-nt3_int(:,idx))/N_trap3/tau_SHR_p_3;
%         radiative_rec = krad *PL(:,idx); 
%         loglog(t, trapping_e, t,detrapping_e, t,trapping_p)
%         hold on
%         if p1_3> n1_3
%             loglog(t, detrapping_p)
%         end
% %         loglog(t,n1_3*nt3_int(:,idx)./(N_trap3-nt3_int(:,idx)),'.-')
%         xlabel('Time t');
%         ylabel('rates dn/dt (cm^{-3} s^{-1})');
%         legend('trapping e','detrapping e','trapping h','detrapping h','Location','best')
%         title(['n(t=0) = ',num2str(nstartarr(idx),'%.E'),' cm^{-3}'])
%             xlim([1e-9 3e-4])
% %         ylim([1e12 1e18])
%     end
%     
%     
%        % recombination rates vs time
%     figure('Units','pixels','Position',[300 100 833 800]);
%         sgtitle({['Recombination rates vs time'],[ '{\fontsize{11}',...
%             'N_{trap} = ',sprintf('%.0e',N_trap3),' cm^{-3}, ',...
%             'n_{1} = ',sprintf('%.0e',n1_3),' cm^{-3}, ',...
%             '\tau_{n} = ',sprintf('%.0e',tau_SHR_n_3),' s, ',...
%             '\tau_{p} = ',sprintf('%.0e',tau_SHR_p_3),' s',...
%             '}']}) 
%     for idx = 1:4
%         subplot(2,2,idx)
%         %         loglog(t, nint, t, pint,t,(nint.*pint).^0.5)
%         %         loglog(t, nint, t, pint)
%         %             loglog(t, nint+nt1_int+nt2_int+nt3_int, t, pint,'--')
%         %             idx=4;
%         trapping_e = nint(:,idx).* (N_trap3-nt3_int(:,idx))/N_trap3/tau_SHR_n_3;
%         detrapping_e = n1_3*nt3_int(:,idx)/N_trap3/tau_SHR_n_3;
%         trapping_p = pint(:,idx).* nt3_int(:,idx)/N_trap3/tau_SHR_p_3;
%         detrapping_p = p1_3*(N_trap3-nt3_int(:,idx))/N_trap3/tau_SHR_p_3;
%         radiative_rec = krad *PL(:,idx); 
%         loglog(t, trapping_e-detrapping_e, t,trapping_p-detrapping_p, t,radiative_rec)
%         hold on
%                 loglog(t,nint(:,idx)/tau_SHR_n_3,'-.')
% %         if p1_3> n1_3
% %             loglog(t, detrapping_p)
% %         end
% %         loglog(t,n1_3*nt3_int(:,idx)./(N_trap3-nt3_int(:,idx)),'.-')
%         xlabel('Time t');
%         ylabel('rates dn/dt (cm^{-3} s^{-1})');
%         legend('net trapping e','net trapping h','radiative rec','Location','best')
%         title(['n(t=0) = ',num2str(nstartarr(idx),'%.E'),' cm^{-3}'])
%             xlim([1e-9 3e-4])
% %         ylim([1e12 1e18])
%     end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
%     
%     % n and p vs Vint (showing path of photodoping)
%     figure
%     %     semilogy(Vint(:,idx),nint(:,idx),Vint(:,idx), pint(:,idx),'--')
%     semilogy(Vint,nint,Vint, pint,'--')
%     xlabel('Fermi-level splitting E_F (eV)');
%     ylabel('charge carrier concentration n (cm^{-3})');
%     xlim([0.95 1.5]);

    
    % EF vs time
%     figure
%     %     semilogy(Vint(:,idx),nint(:,idx),Vint(:,idx), pint(:,idx),'--')
%     loglog(t,Vint)
%     xlabel('time t (s)')
%     ylabel('Fermi-level splitting E_F (eV)');
    %
    % figure(3)
    % subplot(1,2,1)
    % loglog(t, tau,'o', t, tauanal, t, tau2,'*')
    % xlabel('Time t');
    % ylabel('lifetime \tau (s))');
    % % axis([min(t) max(t) max(max(nint))/1e5 max(max(nint))*1.1])
    % subplot(1,2,2)
    % semilogy(t, tau,'o', t, tauanal, t, tau2, '*')
    % % axis([min(t) max(t) max(max(nint))/1e5 max(max(nint))*1.1])
    % xlabel('Time t');
    % ylabel('lifetime \tau (s))');
    
    % figure(4)
    % subplot(1,2,1)
    % loglog(nint, tau,'o', nint, tauanal, nint2, tau2, '*')
    % xlabel('carrier concentration n (cm^{-3})');
    % ylabel('lifetime \tau (s))');
    % % axis([min(t) max(t) max(max(nint))/1e5 max(max(nint))*1.1])
    % subplot(1,2,2)
    % loglog(nint, tau,'o', nint, tauanal, nint2, tau2, '*')
    % % axis([min(t) max(t) max(max(nint))/1e5 max(max(nint))*1.1])
    % xlabel('carrier concentration n (cm^{-3})');
    % ylabel('lifetime \tau (s))');
    %
    %     figure(5)
    %     %subplot(1,2,1)
    %     %     semilogy(Vint, tau,'o', Vint, tauanal, Vint2, tau2, '*', Vint2, tauanal2)
    %     semilogy(Vint, tau,'o', Vint, tauanal)
    %     %  axis([1.0 max(max(Vint)) min(min(tauanal)) 10*max(max(tauanal))])
    %     xlabel('voltage V (V)');
    %     ylabel('lifetime \tau (s))');
    %     legend('tau left','tau_{anal} left')
    %     %     legend('tau left','tau_{anal} left','tau right','tau_{anal} right')
    %     % subplot(1,2,2)
    %     % semilogy(Vint, tau,'o', Vint, tauanal, Vint2, tau2, '*')
    %     % axis([1.0 max(max(Vint)) min(min(tauanal)) max(max(tauanal))])
    %     % xlabel('voltage V (V)');
    %     % ylabel('lifetime \tau (s))');
    %
    %     figure(6)
    %     %subplot(1,2,1)
    %     %     semilogy(Vint, tau,'o', Vint, tauanal, Vint2, tau2, '*', Vint2, tauanal2)
    %     semilogy(Vint, k_diff,'o')
    %     %  axis([1.0 max(max(Vint)) min(min(tauanal)) 10*max(max(tauanal))])
    %     xlabel('voltage V (V)');
    %     ylabel('k_diff (cm3/s))');
    %     legend('tau left','tau_{anal} left')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% big figure
    
    fig = figure('Units','pixels','Position',[300 100 1300 800]);
    sgtitle({['mobility \mu = ',num2str(mu),' cm^2V^{-1}s^{-1}'],[ '{\fontsize{11}',...
            'N_{trap} = ',sprintf('%.0e',N_trap3),' cm^{-3}, ',...
            'n_{1} = ',sprintf('%.0e',n1_3),' cm^{-3}, ',...
            '\tau_{n} = ',sprintf('%.0e',tau_SHR_n_3),' s, ',...
            '\tau_{p} = ',sprintf('%.0e',tau_SHR_p_3),' s',...
            '}']}) 
    % PL vs time
    subplot(2,3,1)
%     loglog(t,(nint.*pint).^0.5,'--')
%     hold on
    loglog(t,PL.^0.5)
    xlabel('Time t');
    ylabel('PL (~cm^{-2}s^{-1}eV^{-1})');
    lg={};
    for i = 1:length(nstartarr)
        lg{i} = ['n(t=0) = ',num2str(nstartarr(i),'%.E'),' cm^{-3}'];
    end
    legend(lg,'FontSize',7,'Location','best')
    xlim([1e-9 3e-4])
    xl_time = xlim;
    PLmax = max(max((PL).^0.5));
    ylim([1e-9*PLmax PLmax])
    
    % tau vs time
    subplot(2,3,2)
    loglog(t,t,'k:')
    hold on
    lg2 = {'tau=t'};
    for i = 1:length(nstartarr)
        plt1 = loglog(t,tau(:,i),'SeriesIndex',i);
%         plt = loglog(t,t'+1./(mean(k_diff(end-5:end,i)).*nstartarr(i)),':','SeriesIndex',i);
        k_ana(i) = 1./(tau_SHR_p_3*(N_trap3+n1_3)+tau_SHR_n_3.*(nstartarr(i))*1); %+n1_3)./2./n1_3);
        plt = loglog(t,t'+1./(k_ana(i).*nstartarr(i)),':','SeriesIndex',i);
        %         plt2 = loglog(t,t'+1./(k_diff(:,i).*nstartarr(i)),'-.','SeriesIndex',i);
        plt2 = loglog(t, tau_n(:,i),'-.','SeriesIndex',i);
        plt3 = loglog(t, tau_p(:,i),'--','SeriesIndex',i);
%         plt4 = loglog(t, nint(:,idx)*tau_SHR_n_3./(nint(:,idx)*beta_p3/beta_n3-n1_3),'x','SeriesIndex',i);
        %         loglog([t_trap_equil(i),t_trap_equil(i)],[(min(tau(:,i))),(max(tau(:,i)))],'--','SeriesIndex',i)
        
        %         lg{3*i-1} = ['tau_{',num2str(nstartarr(i),'%.E'),'}'];
        %         lg{3*i} =['tau_{',num2str(nstartarr(i),'%.E'),'}=t+1/(nk_{diff,end})'];
        %         lg{3*i+1} =['tau_{',num2str(nstartarr(i),'%.E'),'}=t+1/(nk_{diff})'];

    end
%     lg2{2} = ['tau_{',num2str(nstartarr(i),'%.E'),'}'];
%     lg2{3} =['tau_{',num2str(nstartarr(i),'%.E'),'}=t+1/(nk_{diff,end})'];
%     lg2{4} =['tau_{',num2str(nstartarr(i),'%.E'),'}=t+1/(nk_{diff})'];
    lg2{2} = ['\tau_{diff} = 2/(1/\tau_{n}+1/\tau_{p})'];
    lg2{3} =['\tau = t+1/(nk_{diff,end})'];
    lg2{4} =['\tau = \tau_{n}'];
    lg2{5} =['\tau = \tau_{p}'];
    set(gca,'XScale','log')
    set(gca,'YScale','log')
    xlabel('time t(s)');
    ylabel('lifetime \tau (s))');
    legend(lg2{:},'Location','best','FontSize',7)
    xlim(xl_time)
    ylim([1e-9 3e-4]);
    yl_tau = ylim;
    
    
    % kdiff vs time
    subplot(2,3,5)
    for i = 1:length(nstartarr)
        loglog(t,k_diff(:,i),'SeriesIndex',i)
        hold on
        t_trapequil = tau_SHR_n_3*((2*E_trap3-Eg)/2/kT+log(nstartarr(i)/N_trap3*sqrt(Nc/Nv)));
        loglog(t_trapequil*k_diff(:,i)./k_diff(:,i),k_diff(:,i),'--','SeriesIndex',i)
%         k_ana(i) = 1./(tau_SHR_p_3*(N_trap3+n1_3));
        loglog(t,k_ana(i).*t./t,'SeriesIndex',i)
    end
    set(gca,'XScale','log')
    set(gca,'YScale','log')
    xlabel('time t(s)');
    ylabel('recombination coefficient k_{diff} (cm^3s^{-1}))');
    xlim(xl_time)
    %     legend(lg{:},'Location','best')
%     ylim([5e-11 5e-8]);    
%     yl_k = ylim;
    
%     % n/nt vs time
%     subplot(2,3,4)
%     loglog(t,nint./nt1_int,t,ones(length(t),1)*n1_1/N_trap)
%     xlabel('time t(s)')
%     ylabel('n/n_t')
%     lg3 = [lg,'n_1 / N_t'];
%     legend(lg3,'FontSize',7)
%     xlim(xl_time)

        % ALTERNATIVE: ratio (high wavelength low wavelength) vs time
    subplot(2,3,4)
%     loglog(t,nint./nt1_int,t,ones(length(t),1)*n1_1/N_trap)
    semilogx(t,RATIO(:,:,XX+1))
    xlabel('time t(s)')
    ylabel('ratio spectrum (low \lambda / high \lambda)')
%     lg3 = [lg,'n_1 / N_t'];
%     legend(lg3,'FontSize',7)
    xlim(xl_time)
    
    % tau vs Vint / EF
    subplot(2,3,3)
    semilogy(Vint, tau)
    hold on
    x_V = [min(min(Vint)),max(max(Vint))];
    const = @(x)x.*x_V.^0;
    semilogy(x_V,const(tau_SHR_n_3),'-.')
    semilogy(x_V,const(tau_SHR_p_3),'--')
    [~,idx_V_1p4] = min(abs(Vint(:,end)-1.4));
    semilogy(x_V,exp(-x_V/2/kT)/exp(-1.4/2/kT)*tau(idx_V_1p4,end),':')
%     semilogy(Vint, tau_n,':')
%     semilogy(Vint, tau_p,':')
    xlabel('voltage V (V)');
    ylabel('lifetime \tau (s))');
        lg4 = [lg,'\tau_{SRH,n}','\tau_{SRH,p}'];
    legend(lg4,'FontSize',7,'Location','best')
    xlim([0.95 1.5]);
    xl_V = xlim;
    ylim(yl_tau)
    
    % kdiff vs Vint
    subplot(2,3,6)
    semilogy(Vint, k_diff)
    hold on
    semilogy(x_V,const(ktot),'--')
    xlabel('voltage V (V)');
    ylabel('recombination coefficient k_{diff} (cm^3s^{-1}))');
    xlim(xl_V)
%     ylim(yl_k)
    
%     figure
%     hold on
%     for i = 1:length(nstartarr)
%     loglog(t,nint(:,i),t,pint(:,i),':',t,nt3_int(:,i),'--','SeriesIndex',i)
%     end
%          set(gca,'XScale','log')
%     set(gca,'YScale','log')
    
    
    
    % figure(6)
    % loglog(Sarr, avratPL, 'o-')
    %%%% YYadd ratio figure%%%
    %     figure(100)
    %     semilogx(RATIO2(:,1), RATIO2(:,2), 'x:')
    %     % semilogx(RATIO(:,1), RATIO(:,2), data_Ratio(:,1), data_Ratio(:,2),'o')
    %     xlabel('time {\it t} (ns)');
    %     ylabel('Ratio {\it R} ( )');
    %%YY add end%%
    %diffRatio(:,1)=t*1e9;
    %diffRatio(:,XX+1)=gradient(RATIO(:,XX+1),RATIO(:,1));
    % diffRatio(:,2)=diff(RATIO(:,2))/diff(RATIO(:,1));
    
    
     % tau vs Vint
%     figure
%     loglog(pint, tau)
%     hold on
% %     x_V = [min(min(Vint)),max(max(Vint))];
% %     const = @(x)x.*x_V.^0;
% %     semilogy(x_V,const(tau_SHR_n_3),'-.')
% %     semilogy(x_V,const(tau_SHR_p_3),'--')
% %     semilogy(Vint, tau_n,':')
% %     semilogy(Vint, tau_p,':')
% ptest = logspace(14,19);
%     loglog(ptest, tau_SHR_p_3*(n1_3+N_trap3)./ptest)
%     xlabel('p (1/cm3)');
%     ylabel('lifetime \tau (s))');
% %         lg4 = [lg,'\tau_{SRH,n}','\tau_{SRH,p}'];
% %     legend(lg4,'FontSize',7,'Location','best')
% %     xlim([0.95 1.5]);
% %     xl_V = xlim;
% %     ylim(yl_tau)

    
end

% --------------------------------------------------------------------------
% differential equations for n, p, nt1, nt2, nt3
function [c,f,s] = pdex1pde(x,t,u,DuDx)
global Dn Dp taub krad ni beta_n beta_p N_trap e_n e_p beta_n2 beta_p2 N_trap2 e_n2 e_p2 beta_n3 beta_p3 N_trap3 e_n3 e_p3 d
c = [1; 1; 1; 1;1];
f = [Dn; Dp; 0; 0; 0].*DuDx;

%aa=(x>0);
%  aa=(x>400e-7);
% aa=(x<d*0.05);
aa=(x>d*0.95); % defines the right surface
aa2=(x<d*0.05);% defines the left surface
aa=1; % defines the right surface
aa2=1;% defines the left surface
% aa=(x>760e-7); % defines the right surface
% aa2=(x<40e-7);% defines the left surface
%three defect at bulk%
% s = [-(u(1)-ni)/taub-k*(u(1)*u(2)-ni^2)-beta_n*u(1)*(N_trap-u(3))-beta_n2*u(1)*(N_trap2-u(4))-beta_n3*u(1)*(N_trap3-u(5))+e_n*u(3)+e_n2*u(4)+e_n3*u(5) ;  -(u(2)-ni)/taub-k*(u(1)*u(2)-ni^2)-beta_p*u(2)*u(3)-beta_p2*u(2)*u(4)-beta_p3*u(2)*u(5)+e_p*(N_trap-u(3))+e_p2*(N_trap2-u(4))+e_p3*(N_trap3-u(5)) ;  beta_n*u(1)*(N_trap-u(3))-beta_p*u(2)*u(3)-e_n*u(3)+e_p*(N_trap-u(3)); beta_n2*u(1)*(N_trap2-u(4))-beta_p2*u(2)*u(4)-e_n2*u(4)+e_p2*(N_trap2-u(4)); beta_n3*u(1)*(N_trap3-u(5))-beta_p3*u(2)*u(5)-e_n3*u(5)+e_p3*(N_trap2-u(5))];
% bulk one，surface two
s = [-(u(1)-ni)/taub-krad*(u(1)*u(2)-ni^2)-beta_n*u(1)*(N_trap-u(3))-aa*beta_n2*u(1)*(N_trap2-u(4))-aa2*beta_n3*u(1)*(N_trap3-u(5))+e_n*u(3)+aa*e_n2*u(4)+aa2*e_n3*u(5); ...
    -(u(2)-ni)/taub-krad*(u(1)*u(2)-ni^2)-beta_p*u(2)*u(3)-aa*beta_p2*u(2)*u(4)-aa2*beta_p3*u(2)*u(5)+e_p*(N_trap-u(3))+aa*e_p2*(N_trap2-u(4))+aa2*e_p3*(N_trap3-u(5));...
    beta_n*u(1)*(N_trap-u(3))-beta_p*u(2)*u(3)-e_n*u(3)+e_p*(N_trap-u(3)); ...
    aa*beta_n2*u(1)*(N_trap2-u(4))-aa*beta_p2*u(2)*u(4)-aa*e_n2*u(4)+aa*e_p2*(N_trap2-u(4));...
    aa2*beta_n3*u(1)*(N_trap3-u(5))-aa2*beta_p3*u(2)*u(5)-aa2*e_n3*u(5)+aa2*e_p3*(N_trap3-u(5))];
%surface three
% s = [-(u(1)-ni)/taub-k*(u(1)*u(2)-ni^2)-aa*beta_n*u(1)*(N_trap-u(3))-aa*beta_n2*u(1)*(N_trap2-u(4))-aa*beta_n3*u(1)*(N_trap3-u(5))+aa*e_n*u(3)+aa*e_n2*u(4)+aa*e_n2*u(5);  -(u(2)-ni)/taub-k*(u(1)*u(2)-ni^2)-aa*beta_p*u(2)*u(3)-aa*beta_p2*u(2)*u(4)-aa*beta_p3*u(2)*u(5)+aa*e_p*(N_trap-u(3))+aa*e_p2*(N_trap2-u(4))+aa*e_p3*(N_trap3-u(5));  aa*beta_n*u(1)*(N_trap-u(3))-aa*beta_p*u(2)*u(3)-aa*e_n*u(3)+aa*e_p*(N_trap-u(3));  aa*beta_n2*u(1)*(N_trap2-u(4))-aa*beta_p2*u(2)*u(4)-aa*e_n2*u(4)+aa*e_p2*(N_trap2-u(4)); aa*beta_n3*u(1)*(N_trap3-u(5))-aa*beta_p3*u(2)*u(5)-aa*e_n3*u(5)+aa*e_p3*(N_trap3-u(5))];
%first trap is shallow
%s = [-(u(1)-ni)/taub-k*(u(1)*u(2)-ni^2)-aa*beta_n*u(1)*(N_trap-u(3))-beta_n2*u(1)*(N_trap2-u(4))-beta_n3*u(1)*(N_trap3-u(5))+e_n*u(3)+e_n2*u(4)+e_n2*u(5);  -(u(2)-ni)/taub-k*(u(1)*u(2)-ni^2)-aa*beta_p*u(2)*u(3)-beta_p2*u(2)*u(4)-beta_p3*u(2)*u(5)+aa*e_p*(N_trap-u(3))+e_p2*(N_trap2-u(4))+e_p3*(N_trap3-u(5));  aa*beta_n*u(1)*(N_trap-u(3))-aa*beta_p*u(2)*u(3)-aa*e_n*u(3)+aa*e_p*(N_trap-u(3));  beta_n2*u(1)*(N_trap2-u(4))-beta_p2*u(2)*u(4)-e_n2*u(4)+e_p2*(N_trap2-u(4)); beta_n3*u(1)*(N_trap3-u(5))-beta_p3*u(2)*u(5)-e_n3*u(5)+e_p3*(N_trap3-u(5))];

end

% charge distribution at time 0 when illuminated from left
function u0 = pdex1ic(x)
global alpha d nint0 n0 p0 ntrap0_1 ntrap0_2 ntrap0_3
u00 = alpha*d*nint0/(1-exp(-alpha*d));
u0 = [u00*exp(-alpha*x)+n0; u00*exp(-alpha*x)+p0;ntrap0_1; ntrap0_2; ntrap0_3];
% u00*Dn*alpha^2;

end

% charge distribution at time 0 when illuminated from right
function u0 = pdex1ic2(x)
global alpha d nint0 n0 p0 ntrap0_1 ntrap0_2 ntrap0_3
% u0 = 1e18*ones(1,length(x));
u00 = alpha*d*nint0/(1-exp(-alpha*d)); %alpha and d factor for normalization 
u0 = [u00*exp(-alpha*(d-x))+n0; u00*exp(-alpha*(d-x))+p0;ntrap0_1; ntrap0_2; ntrap0_3];
end
% %--------------------------------------------------------------------------

% boundary conditions?
function [pl,ql,pr,qr] = pdex1bc(xl,ul,xr,ur,t)
global Sl Sr ni
pl = [-Sl*(ul(1)-ni)/2; -Sl*(ul(2)-ni)/2; 0; 0; 0];%%%%note Factor 2
ql = [1;1;1;1;1];
pr = [Sr*(ur(1)-ni)/2; Sr*(ur(2)-ni)/2; 0; 0; 0];
qr = [1;1;1;1;1];
end


