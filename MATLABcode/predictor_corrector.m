%exner��������Active layer�̎��ł̎擾_161005
function A_2nd_next = predictor_corrector(A_2nd, params, CP_Ci, Es_i)
  %% A_2nd��(���a�K��*2)�sn��̔z��Alm�̓�t/��x�Adt�̓�t
  %��A/��t + ��*��B1/��x + ��B2/��x = Z���v�Z���āA���̃^�C���X�e�b�v��A_next���o�͂���

%   get_Z_2nd; %A����Z���Z�o����֐�
  %% ��{�p�����[�^�[���擾
  dt = params.dt;
  %% A����Z���Z�o����
  Z_2nd = get_Z_2nd(A_2nd, params, CP_Ci, Es_i);
  %% �I�C���[�@��p���ė\���qfp���v�Z����
  A_predictor = A_2nd  + dt .* Z_2nd;
  Z_2nd_star = get_Z_2nd(A_predictor, params, CP_Ci, Es_i); %���Ɏg��Z���Z�o
  %% �\���q�E�C���q�ŏI�X�e�b�v���v�Z����
  A_2nd_next = A_2nd  + 0.5 .* dt .* (Z_2nd + Z_2nd_star);
  
end