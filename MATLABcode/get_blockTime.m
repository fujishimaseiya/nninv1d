   function [blockTime] = get_blockTime(sumHi, time, para)
   %% �͐ϕ��̌�����Ԃ�����ɂ����鎞�Ԃ̎Z�o
   
   % division�̐ݒ�
    temp = sumHi;
    divi = para.divi;
    range= 0:divi;
    d_sumHi = sumHi(end) / divi;
    Hi_range = (range * d_sumHi).';

    % �픍�����͍폜
    L = sumHi > sumHi(end); %�_���z��ɂ��C�ŏI�͐ϕ�����ʂ̑͐ϕ��͍폜
    time(L) = [];
    sumHi(L) = [];
    
    % �ɂ߂ď����������͍폜 (�����n�`�i0�j�̕����͍폜���܂ށj
    L = sumHi < 0.0001; %�_���z��ɂ��C���ɏ������l�͍폜
    time(L) = [];
    sumHi(L) = [];
    
    time_temp = zeros(size(time,1)+1,1);%1�s�ڂ�0�b��������
    time_temp(2:end) = time; %0�b�̌��time��������
    sumHi_temp = zeros(size(sumHi,1)+1,1);%1�s�ڂ�0�b��������
    sumHi_temp(2:end) = sumHi; %0�b�̌��time��������
    time = time_temp;
    sumHi = sumHi_temp;
    
    % ���ԕ��
    if sumHi(end) ~= 0
        blockTime = interp1(sumHi, time, Hi_range); %vq = interp1(x,v,xq) 1 ���f�[�^���} (�e�[�u�� ���b�N�A�b�v)
    else 
        blockTime = 0;
    end
    
   end