%���[h�̕ω����ɉ����ăV���b�N�z���X�L�[����K�p_160204
function A_next = Jameson(h, A, params)
  %% ���[h�̕ω����ɉ����ăV���b�N�z���X�L�[����K�p���AA���C������
  
  %% ��{�p�����[�^
  lm = params.lm;
  gnum = size(A,2);%�O���b�h��
  dx_dt = params.dx ./ params.dt; % ��x/��t
  kappa = params.kappa;%�l�דI�g�U�W��
  
  A_next = zeros(size(A)); %�v�Z���ʊi�[�p
  eps_i = zeros(1,gnum);
  eps_i_half = zeros(1,gnum);
  
  %% �V���b�N�z���p�̊g�U�W�����v�Z
  eps_i(2:gnum-1) = abs(h(3:gnum) - 2 .* h(2:gnum-1) + h(1:gnum-2)) ./ (abs(h(3:gnum)) + 2 .* abs(h(2:gnum-1)) + abs(h(1:gnum-2)));  
  
  %% �V���b�N�z���p�̊g�U�W���i�O���b�h�����炵�j���v�Z
  eps_i_half(2:gnum-1) = kappa .* dx_dt .* max([eps_i(3:gnum),eps_i(2:gnum-1)]);
  
  %% �V���b�N�z���p�̐��l�S���X�L�[����K�p
  for k = 1:size(A,1)
    A_next(k,1) = A(k,1);
    A_next(k,2:end-1) = A(k,2:end-1) + eps_i_half(2:end-1) .* (A(k,3:end) - A(k,2:end-1)) - eps_i_half(1:end-2) .* (A(k,2:end-1) - A(k,1:end-2));
    A_next(k,gnum) = A(k,gnum) - eps_i_half(gnum-1) .* (A(k,gnum) - A(k,gnum-1));
  end
  
end