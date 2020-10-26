%���������̐ݒ�_160204
function [A, A_2nd, params] = set_init(setfile)
%TurbSurge�𗬂�鍬�����̐��l�v�Z���s�����߂̃p�����[�^��ݒ肷��
    calcTurbSurge %setfile�̓ǂݍ���

    %A(1,:) ����̌����ih�j
    %A(2,:) �P�ʋ���������̉^���ʁiU*h�j
    %A(3,:) �P�ʋ���������̑͐ϕ��̗ʁiCi*h�j
    %A(5,:) �͐ϕ��̌��� �ieta_sand�j2016/05/18
    %A(7,:) �͐ϕ��̗��x���z �iFsand�j2016/05/18
    %setfile �ݒ�t�@�C��
    %% �v�Z�O���b�h�ƒn�`����
    [xo, eta_init] = make_topo(topofile, topodx);%�t�@�C������n�`�f�[�^��ǂݍ���
    params.grid = nogrid;%��ԃO���b�h��
    params.xo = xo;%�������W
    params.noGS = noGS; %���a�K��
    %% �㗬�[�̋��E����
    params.ho = ho;
    params.lo = lo;
    params.Ci_init = Ci_init;%n���a�Ɋg��
    params.Co = sum(Ci_init); %�g�[�^���̏����Z�x
    params.Uo = 1.2 * sqrt(R * g * params.Co * params.ho); %naruse 
    params.Fi = Fi_init; %���x���z
    params.La = La; %�����w�̌����@(Arai,2011MS) %2016/05/17 add
    %% �v�Z�����󂯓n���p�ϐ�
    params.eta_init = eta_init;%�����n�`
    params.eta = eta_init;%���݂̒n�`
    params.topodx = topodx;%�n�`�̃O���b�h�Ԋu�@%�����W�ł̕�ԁi��j�Ԋu
    params.dx = 1 ./ (nogrid - 1);%��ԃO���b�h�Ԋu
    params.x = (0:params.dx:1);%��ԃO���b�h
    params.s = params.lo ./ params.ho;%�w�b�h�́i�������j�ʒu
    params.sdot = 1; %�w�b�h�̈ړ����x
    %params.stable_mlevel = stable_mlevel;%�����[�ɂ�����D�����̍ő卂��
    %% ���̂ق��̃p�����[�^�[
    params.p = p;%�͐ϕ������グ�}���ϐ�
    params.R = R;%�͐ϕ�������d
    params.Di = Di;%�͐ϕ��̗��a
    params.nu = nu;%���̓��S���W��
    params.vs = get_vs(params.R, params.Di, params.nu) / params.Uo;%���̖��������~���x
    %params.S = get_S(params.dx, params.eta);%�����ΖʌX��, set_params�Ɉڍs
    params.g = g;%�d�͉����x
    params.Cf = Cf;%��ʒ�R�W��
    params.ro = ro;%��ʋߖT�^���ϔZ�x��
    params.lambdap = lambdap;%�͐ϕ��Ԍ���
    params.t = 0;%�v�Z�@���̎���
    params.kappa = kappa;%���l�S��
    params.Rp = sqrt(R .* g .* Di) .* Di ./ nu;%���q���C�m���Y��
    params.Rio = R .* g .* params.Co .* params.ho ./ params.Uo .^2;%�������`���[�h�\����

    params.prefix = prefix; %�t�@�C����ۑ�����t�H���_��
    params.topofile = topofile; %�n�`�f�[�^��ۑ�����t�@�C����
    params.dt = 0;
    params.eps = 1 .* 10 ^-6;%�ɂ߂ď�������
    %% ����̏�������
    params.noCP = size(params.eta_init,2); %control points(��_) 2016/11/15
    A = set_init_flow(params);
    A_2nd = set_init_layer(params);
    params.etai_init = A_2nd(1:noGS,:);
    params = set_params(A, params);%���I�p�����[�^�̃Z�b�g
    %delete([prefix 'time.txt']);
end
