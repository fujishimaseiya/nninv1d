%exner��������Active layer�̎��ł̎擾_161005
function Z_2nd = get_Z_2nd(A_2nd, params, Ci, Es_i)
  %% A����A_2nd�iExner��Active Layer�j���Z�o����֐�
    noGS = params.noGS;%���a�K
    noCP = params.noCP;
    vs = params.vs;%�͐ϕ����~���x
    
    La = params.La;
    ro = params.ro;%��ʁ^���ϔZ�x��
    ho = params.ho;
    Uo = params.Uo;
    lambdap = params.lambdap; %�͐ϕ��Ԍ���
    
    Fi = A_2nd(noGS+1:2 * noGS,:);%�Œ���W�n�ł̗��x���z
    
    etai = A_2nd(1:noGS,:);%Exner���������Z�o�����n�`
    etai_init = params.etai_init;
    L = etai < etai_init;%L = eta_i < repmat(eta_local, noGS, 1);%�_���z��ɂ�錟�o, �N�H�ʂ��傫�������ꍇ�ɂ͕ύX, �����i���j�͂����Ă���   
    Es_i(L) = ro * Ci(L) ./ Fi(L);
%     Ct = sum(Ci,1);
%     dep_i = (Uo * repmat(vs, 1, noCP)) .* (ro .* Ci - Fi .* Es_i); %Exner�������ł͕������t�ł���̂ŁC���̂܂� dep_i = repmat(vs, 1, noCP) .*  (ro * Ci - Fi .* Es_i./ Co);
    dep_i =  (ho * Uo * repmat(vs, 1, noCP)) .* (ro * Ci - Fi .* Es_i);
    %�g�[�^���ŐN�H���N�������Ȃ����߂̏���
    eroded_area = dep_i < 0;%�N�H���������Ă���̈�̌��o
    if dep_i < 0 
        a = 1;
    end
    dep_i(eroded_area) = 0;%A(5,:) ���͐ϕ��̌��� �ieta_sand�j

  %% �Œ���W�n�ɂ�����\�[�X��
    Z_2nd(1:noGS,:)= dep_i ./ (1 - lambdap); %Exner
    eta_t = sum(dep_i, 1) ./ (1 - lambdap);
    Z_2nd(noGS+1:2 * noGS, :) = (Z_2nd(1:noGS,:) ./ La - Fi ./ La .* repmat(eta_t, noGS, 1)); %Active Layer; eta_i ./ params.La - Fi .* eta_total
end

% memo
% dep = - dep;%Exner�������ł͕������t
% Z_2nd(noGS+1:2 * noGS, :) = (Z_2nd(1:noGS,:) ./ La - Fi ./ La .* repmat(eta_t, noGS, 1));