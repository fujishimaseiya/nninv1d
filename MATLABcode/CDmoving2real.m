%% 2016/11/15
function [CP_Ci, CP_Es_i] = CDmoving2real(A, params, Es_i)
    %% �p�����[�^���o
    noGS = params.noGS; %number of grid
    grid = params.grid;
    x = params.x .* params.s .* params.ho;%����Ԃւ̕ϊ�

    h_temp = A(1,:);
    Ci_temp = A(3:2 + noGS,:) ./ repmat(h_temp, noGS, 1);
    %% ����Ԃɖ߂�
    if size(params.Ci_init, 2) == 1 %1��̏ꍇ��FM�ł���]�u�Ȃ��C�񂪕�������ꍇ��ga�ł̏����l�����ɂ��]�u
        Ci = Ci_temp .* repmat(params.Ci_init, 1, grid);
    else
        Ci = Ci_temp .* repmat(params.Ci_init.', 1, grid); %�]�u
    end
    
    %% Control point
    CP_Ci = zeros(noGS, size(params.eta,2));
    CP_Es_i = zeros(noGS, size(params.eta,2));
    Ci(isnan(Ci)) = 0;
    Es_i(isnan(Es_i)) = 0;
    for k= 1:noGS
        CP_Ci(k,:) = interp1(x, Ci(k,:), params.xo, 'pchip', 0);% ���a�K�̑͐ϕ��̌��� control point
        CP_Es_i(k,:) = interp1(x, Es_i(k,:),params.xo, 'pchip', 0);
    end
end

%% memo

%     U_temp = A(2,:) ./ h_temp; %A��1,2�s�ڂ���U����肾��
%     logical = U_temp  < eps;
%     U_temp (logical) = eps;%�����[���̒n�_��eps�Œu�������� U(find(U < eps)) = eps;
%     U = U_temp .* params.Uo;