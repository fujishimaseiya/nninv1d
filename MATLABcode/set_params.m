%�p�����[�^�̐ݒ�_160204
function [newparams] = set_params(A, params)
    %% ���̃^�C���X�e�b�v�Ɍ����Čv�Z�p�p�����[�^�[���X�V����֐�
    
    %A(1,:) ����̌����ih�j
    %A(2,:) �P�ʋ���������̉^���ʁiU*h�j
    
%     set_dt;
%     get_S;
    
    newparams = params;
    newparams.sdot = A(2,end) ./ A(1,end);%�w�b�h�̈ړ����x���v�Z�̃^�C���X�e�b�v�ł̃w�b�h�̈ʒu���v�Z
    newparams.s = params.s + newparams.sdot .* params.dt; %�w�b�h�̈ʒu
    [newparams.dt, newparams.lm] = set_dt(A,params); %���ԃX�e�b�v��ݒ肷��
    newparams.slope = get_S(params); %�Ζʌ��z���Z�o����

end