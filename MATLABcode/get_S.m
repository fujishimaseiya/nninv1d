%% S�i�X�΁j�̎擾
function slope = get_S(params)
  %S���Z�o����֐�
  
  x_real = params.x .* params.s .* params.ho;%����Ԃł̃O���b�h���W���擾 %
  slope = zeros(size(x_real));%�Ζʌ��z���i�[����z��
  eta = params.eta;%�n�`����
  
  %����Ԃł̎Ζʌ��z���v�Z
  S = zeros(1,size(eta,2)); 
%   eta = params.eta;
  topodx = params.topodx;
  gnum = size(S,2);
  Sx = [0:topodx:(gnum - 1) * topodx];
  
  for i = 2:gnum-1
     S(i) = - ( ( eta(i+1) - eta(i-1) ) / ( 2 .* topodx ) ); %�Ζʌ��z
  end
  S(1) = - ( (eta(3) - eta(1)) / ( 2 .* topodx ) ); %�㗬�[�̎Ζʌ��z��^����
  S(gnum) = - ( (eta(gnum) - eta(gnum-2)) / ( 2 .* topodx )); %�����[�̎Ζʌ��z��^����
  
  %���`�⊮�ɂ��v�Z�O���b�h�_�ł̎ΖʌX�΂����߂�
  slope = interp1(Sx,S,x_real);
  
end