%% A�̏㗬�[�ɂ����鋫�E����_160204
function A_upbound = get_A_upbound(A)
  %A�̏㗬�[�ɂ����鋫�E������^����֐�
  %A_upbound = ones(2 + params.noGS,1);
  A_upbound(:,1) = A(:, 2); %�m�C�}���̋��E����
  A_upbound(2,1) = 0; %�Œ�
  
  
%% lock exchange model 
%   A_upbound(2,:) = eps;%zeros;%������lock exchange�@2016/11/08
  
end

%% memo
% %     %h = -local_topo ./ params.ho; % h��h^, �܂�(1 X nogrid)�̍s��
% %     %A = [h;h;h;h;zeros(1,params.grid);-h;(params.Fsand * (1-(h ./ params.La)));(params.Fmud * (1-(h ./ params.La)))] .* A; %h�͖�����������Ă�
% %   A_upbound = [1;1;1; params.eta(1,1) .* params.Fsand ./ params.ho;params.Fsand] .* A_upbound;




