%B�̎擾_160204
function B2 = get_B2(A, params)
  %% A����B���Z�o����֐�
  
  Rio = params.Rio; %�������`���[�h�\����
  Ci_init = params.Ci_init;%�e���a�̑͐ϕ������Z�x
  Co = params.Co;%�͐ϕ������Z�x
  noGS = params.noGS;
  Grid = params.grid;
  B2 = zeros(size(A));

%% 2016/05/17
  h = A(1,:); %A��1�s�ڂ���h����肾��
  U = A(2,:) ./ h; %A��1,2�s�ڂ���U����肾��
  logical = U < eps;
  U(logical) = eps;%�����[���̒n�_��eps�Œu�������� U(find(U < eps)) = eps;
  Ci = A(3:2 + noGS,:) ./ repmat(h, noGS, 1); %A��3�s�ڂ���C_sand����肾��Ci = A(3,:) ./ h;
  Ct = sum(Ci .* repmat(Ci_init, 1, Grid),1) ./ Co; %�g�[�^���̔Z�x
%   Ct = sum(Ci,1); %C_total = C_sand + C_mud; %�g�[�^���̔Z�x
%% �ړ����W�n�ɂ�����ڗ���
  B2(1,:) = U .* h  ; %- A(2,:);
  B2(2,:) = (U .* U .* h + (Rio .* Ct .* h .* h) ./ 2) ;% + Rio .* C_total .* Co .* eta_total)); %- (- (U .* U) .* h - Rio .* C_total .* h .^2 ./ 2);% + Rio .* C_total .* Co .* eta_total));
  B2(3:2 + noGS,:) = Ci .* repmat(h, noGS, 1) .* repmat(U, noGS, 1) ; %- (C_sand .* h .* ( - U));
end

  
