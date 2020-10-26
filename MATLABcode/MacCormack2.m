%�}�N�}�b�N�@_160204
function [A_next, Es_i_next]= MacCormack2(A, A_2nd, params)
  %% A��8�sn��̔z��Alm�̓�t/��x�Adt�̓�t
  %��A/��t + ��*��B1/��x + ��B2/��x = Z���v�Z���āA���̃^�C���X�e�b�v��A_next���o�͂���
  
%   get_B1,B2; %A����B1,B2���Z�o����֐�
%   get_Z; %A����Z���Z�o����֐�
%   get_A_upbound; %A�̏㗬�[�̋��E������^����֐��i3�s1��j
%   get_A_downbound_3; %A�̏㗬�[�̋��E������^����֐��i3�s1��j
%   Jameson;%�V���b�N�z���X�L�[����K�p����֐�
  
  %% ��{�p�����[�^�[���擾
  lm = params.lm; % ��t/��x
  dt = params.dt; % ��t
  alpha = - params.sdot .* params.x ./ params.s; 
  alpha = repmat(alpha, [size(A,1),1]); %alpha = repmat(alpha,[8,1]);
  s = params.s ;
  %% Grid�����Z�o����
  gnum = size(A,2);
  %% A����B1��B2,Z���Z�o����
  B1 = get_B1(A,params);
  B2 = get_B2(A,params);
  
  [Z, Es_i] = get_Z(A, A_2nd, params);
  %% �}�R�[�}�b�N�@���X�e�b�v�v�Z���ʂ��i�[����z��
  A_star = zeros(size(A));
  B1_star = zeros(size(B1));
  B2_star = zeros(size(B2));
  Z_star = zeros(size(Z));
  %% �}�R�[�}�b�N�@��Q�X�e�b�v�v�Z���ʂ��i�[����z��
  A_dstar = zeros(size(A));
  %% �ŏI�v�Z���ʂ��i�[����z��
  A_next = zeros(size(A));
  %% �}�R�[�}�b�N��1�X�e�b�v���v�Z����
%   test1= alpha(:,3:end); %���ς�������H
  test1= (alpha(:,3:end) + alpha(:,2:end-1)) ./ 2; 
  A_star(:,2:end-1) = A(:,2:end-1) - lm .* (test1 .* (B1(:,3:end) - B1(:,2:end-1)) + (B2(:,3:end) - B2(:,2:end-1)) ./ s)  + dt .* Z(:,2:end-1);
%   A_star(:,2:end-1) = A(:,2:end-1) - lm .* (test1 .* (B1(:,3:end) - B1(:,2:end-1)) + (B2(:,3:end) - B2(:,2:end-1))) ./ params.s + dt .* Z(:,2:end-1);
  A_star(:,1) = get_A_upbound(A_star); %A�̏㗬�[���E������^����
  A_star(:,gnum) = get_A_downbound_4(A, params); %A�̉����[���E������^����
  B1_star = get_B1(A_star, params); %���Ɏg��B1���Z�o
  B2_star = get_B2(A_star, params); %���Ɏg��B2���Z�o
  
  [Z_star, Es_i_star] = get_Z(A_star, A_2nd, params); %���Ɏg��Z���Z�o

  %% �}�R�[�}�b�N��2�X�e�b�v���v�Z����
%   test2= alpha(:,2:end-1);
  test2= (alpha(:,2:end-1) + alpha(:,1:end-2)) ./ 2;
  A_dstar(:,2:end-1) = A_star(:,2:end-1) - lm .* (test2 .* (B1_star(:,2:end-1) - B1_star(:,1:end-2)) + (B2_star(:,2:end-1) - B2_star(:,1:end-2)) ./ s) + dt .* Z_star(:,2:end-1);
%   A_dstar(:,2:end-1) = A_star(:,2:end-1) - lm .* (test2 .* (B1_star(:,2:end-1) - B1_star(:,1:end-2)) + (B2_star(:,2:end-1) - B2_star(:,1:end-2)))./ params.s + dt .* Z_star(:,2:end-1);
  A_dstar(:,1) = get_A_upbound(A_dstar); %A�̏㗬�[���E������^����
  A_dstar(:,end) = get_A_downbound_4(A_star, params); %A�̏㗬�[���E������^����

  %% �}�R�[�}�b�N�ŏI�X�e�b�v���v�Z����
  A_next = 0.5 .* (A + A_dstar);
  %% ���E������^����
  A_next = Jameson(A_next(1,:), A_next, params); %�V���b�N�z���p�̐��l�S���X�L�[����K�p����
    
  %% Exner�������Ɏ󂯓n���͐ρE�N�H�̃p�����[�^
  Es_i_next = 0.5 .* (Es_i + Es_i_star); %�S�͐ϕ������Z�x
  A_next(A_next<0) = 0;
end
