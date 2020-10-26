%Z�̏���_160204
function [Z, Es_i] = get_Z(A, A_2nd, params)
  %% A����Z���Z�o����֐�
  
  Rio = params.Rio; %�������`���[�h�\����
  Cf = params.Cf; %��ʒ�R�W��
  ro = params.ro;%��ʁ^���ϔZ�x��
  Ci_init = params.Ci_init;%�e���a�̑͐ϕ������Z�x
  Co = params.Co;%�͐ϕ������Z�x
  vs = params.vs;%�͐ϕ����~���x
  slope = params.slope;%�Ζʌ��z
  noGS = params.noGS;
  Grid = params.grid;
  
  Z = zeros(size(A));
  q = A(2,:); %�������^����
  
  % 2016/05/17
  h = A(1,:); %A��1�s�ڂ���h����肾�� %����������
  U = A(2,:) ./ h; %A��1,2�s�ڂ���U����肾��
  L = U < eps;
  U(L) = eps;%�����[���̒n�_��eps�Œu�������� U(find(U < eps)) = eps;
  Ci = A(3:2 + noGS,:) ./ repmat(h, noGS, 1);%A��3�`���s�ڂ���Ci����肾��
  Ct = sum(Ci .* repmat(Ci_init, 1, Grid),1) ./ Co; %�g�[�^���̔Z�x C_total = C_sand + C_mud;�@%��̘a

  %% �N�H�E�͐ϗʂ̌v�Z
  Fi = A_2nd(noGS+1: 2 * noGS,:); %Fi = A(7,:);
  Es_i = get_Es(A, params);%���̘A�s�W��
  Ci_init_grid = repmat(Ci_init, 1, Grid);
%   dep_i =  repmat(vs, 1, Grid) .* (ro * Ci .* Ci_init_grid - Fi .* Es_i); %���̑͐ϗ� dep_i =  repmat(vs, 1, Grid) .* (ro * Ci - Fi .* Es_i./ Co);
  dep_i =  repmat(vs, 1, Grid) .* (ro * Ci - Fi .* Es_i./ Ci_init_grid); %���̑͐ϗ� dep_i =  repmat(vs, 1, Grid) .* (ro * Ci - Fi .* Es_i./ Co);
    
  %�g�[�^���ŐN�H���N�������Ȃ����߂̏���
  eroded_area = dep_i < 0;%�N�H���������Ă���̈�̌��o
  dep_i(eroded_area) = 0;%A(5,:) ���͐ϕ��̌��� �ieta_sand�j
  Es_i(eroded_area) = ro .* Ci(eroded_area) .* Ci_init_grid(eroded_area) ./ Fi(eroded_area);

  %% �ړ����W�n�ɂ�����\�[�X��
  scrit = q > params.eps; %find(q > params.eps);
  Z(1,scrit) = 0.00153 .* U(scrit) ./ (0.0204 + (Rio .* Ct(scrit) .* h(scrit)) ./ U(scrit) ./ U(scrit));%Z(1,scrit) = 0.00153 .* (q(scrit) ./ h(scrit)) ./ (0.0204 + Rio .* C_total(scrit) .* h(scrit)) ./ (U(scrit) .* U(scrit));
  Z(2,:) = (Rio .* Ct .* h .* slope) - Cf .* (U .* U);
  Z(3:2 + noGS,:) = - dep_i; % - dep_sand; % vs .* (es_sand ./ C_sand_init - ro .* C_sand);

end