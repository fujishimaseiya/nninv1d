%�e��t�@�C����
    prefix = 'test_output\'; %�f�[�^���L�^����t�H���_��
    prefix2 = 'test_input\'; %�f�[�^���L�^����t�H���_��
    topofile = 'test_input\InitTopo.txt'; %�n�`�f�[�^���L�����t�@�C��
    delete('\test_output\*.txt');

%% ���߂������������@inital
    ho_max = 800;
    ho_min = 100;
    lo_max = 800;
    lo_min = 100;
    C_max = 0.01;
    C_min = 0.0001;
    S_min = 0.000;
    S_max = 0.005;
    ho = 1.367199999999999989e+02;
    lo = 5.561299999999999955e+02;
    S = 2.241400000000000163e-03;
    delete(topofile);
    topo = [0,0;5000,-500;50000,-500-S*(50000-5000)];
    dlmwrite(topofile, topo);
    noGS = 3; %���a�K��

    str_noGS = num2str(noGS);
    Di = load([prefix2 ['InitD' str_noGS] '.txt']); %�͐ϕ��̗��a(m) [1000 * 10^-6; 250 * 10^-6; 62.5 * 10^-6];%3���a
    Di = Di .* 10^-3; %���a�̎����𑵂���

%     Ci_init = repmat(0.01 ./ noGS, noGS, 1);%Ci_init = repmat(C_unit, noGS, 1); %�����Z�x�i���j1%�͔Z�� [0.01; 0.01] 
    Ci_init = [2.629600000000000166e-03;1.891099999999999942e-03;2.310299999999999940e-03];
    F_base = 1 ./ noGS;
    Fi_init = repmat(F_base, noGS, 1);% 1-sum(repmat(F_base, noGS-1, 1))]; %GS in Active layer
%% ����̏�������
    nogrid = 50; %��ԃO���b�h��
    topodx = 5;%50; %�n�`�̃O���b�h�Ԋu
    R = 1.65; %�͐ϕ�������d for natural quartz.
    nu = 1.0 * 10^-6; %���̓��S���W��
    g = 9.81; %�d�͉����x(m/s^2)
    Cf = 0.0069; %��ʒ�R�W�� 0.0069 0.004
    ro = 1.5; %��ʋߖT�^���ϔZ�x��
    lambdap = 0.4; %�͐ϕ��Ԍ���
    kappa = 0.001;%0.00001; %���l�S��
    p = 0.1;%�͐ϕ������グ�}��
    La = 0.003; %�����w�̌����@(Arai,2011MS)
    
    %���������̋L�^
    dlmwrite([prefix 'initial_conditions.txt'], [ho, lo, Ci_init(1), Ci_init(2), Ci_init(3), S], '-append');