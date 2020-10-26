%A�̉����[�ɂ����鋫�E����_160204
function A_downbound = get_A_downbound_4(A, params)
    %A�̉����[�ɂ����鋫�E������^����
    A_downbound = zeros(size(A,1),1);
    gnum = size(A,2);
    noGS= params.noGS;
    Rio = params.Rio;
    Ci_init = params.Ci_init;%�e���a�̑͐ϕ������Z�x
    Co = params.Co;%�͐ϕ������Z�x
    Frd_const = 1.2; %�w�b�h�̃t���[�h�����K��1.2�ɂȂ�悤�ɂ��� Huppert and Simpson 1980

    if 0== isreal(params.s) %0�͕��f��
        return
    end

    %% A�̉����[�̋��E������ύX����
    %eta = interp1([0:params.topodx:(size(params.eta,2) - 1) .* params.topodx],params.eta,x); 
    A_downbound(:,1) = A(:,end-1);
    H = A_downbound(1,1);%�O�̗���̌�����ۑ� %H = A_downbound(1,1) * params.ho;
    U = A_downbound(2,1)./A_downbound(1,1); %U = A_downbound(2,1)./A_downbound(1,1) * params.Uo;
    Ci = A_downbound(3:2 + noGS,1) ./ A_downbound(1,1);
    Ct = sum(Ci .* Ci_init,1) ./ Co; %C = sum(A_downbound(3:2 + noGS,1))./A_downbound(1,1).* params.Co; % C = (A_downbound(3,1) + A_downbound(4,1))./h;
    
    A_downbound(1,1) = (U .^2 .* H .^2 ./ (Frd_const .^2 .* Rio .* Ct)).^(1/3); %A_downbound(1,1) = (U .^2 .* h .^2 ./ (Frd_const .^2 .* R .* g .* C)).^(1/3);
    A_downbound(2,1) =  (Frd_const .^2 * Rio .* (Ct .* U .* H) ).^(1/3); %A_downbound(2,1) =  (Frd_const .^2 * (R .* g .* C .* U .* H) ).^(1/3);
    A_downbound(3:2 + noGS,1) = A_downbound(3:2 + noGS,1); %���͐ϕ��Z�x
    
    % confirm Frd
    Frd = A_downbound(2,1) ./ sqrt(Rio .* Ct .* A_downbound(1,1));
    a = 0;
    if (Frd - 1.2)^2 > 0.000000001
        a = a +1;
    end
end

%%�@����
%     A_downbound(1,1) = (Frd_const .^ 2 .* R .* g .* A_downbound(2,1) ./ (A_downbound(3,1) + A_downbound(4,1))) .^ (1/3); %����?�Ȃ������ꂪ����
%     A_downbound(1,1) = A_downbound(2,1) .^ 2 ./ Frd_const .^ 2 ./ R ./ g ./ (A_downbound(3,1) + A_downbound(4,1));%(U1*h1)^2/(Fr^2*R*g*C)