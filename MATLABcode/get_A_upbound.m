%% Aの上流端における境界条件_160204
function A_upbound = get_A_upbound(A)
  %Aの上流端における境界条件を与える関数
  %A_upbound = ones(2 + params.noGS,1);
  A_upbound(:,1) = A(:, 2); %ノイマンの境界条件
  A_upbound(2,1) = 0; %固定
  
  
%% lock exchange model 
%   A_upbound(2,:) = eps;%zeros;%流速はlock exchange　2016/11/08
  
end

%% memo
% %     %h = -local_topo ./ params.ho; % hはh^, また(1 X nogrid)の行列
% %     %A = [h;h;h;h;zeros(1,params.grid);-h;(params.Fsand * (1-(h ./ params.La)));(params.Fmud * (1-(h ./ params.La)))] .* A; %hは無次元化されてる
% %   A_upbound = [1;1;1; params.eta(1,1) .* params.Fsand ./ params.ho;params.Fsand] .* A_upbound;




