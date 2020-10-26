function [dt, lm] = set_dt(A, params) 
    %% 2016/06/28�@�ڗ����x���ɑΉ������N�[�������ɕύX
    
      dx = params.dx; %��ԃO���b�h�Ԋu
      g = params.g; 
      h = A(1,:);
%       alpha = params.sdot .* params.x ; %params.sdot .* params.x ./ params.s ; %get out the minus from MacCormack2
      alpha = - params.sdot .* params.x ./ params.s ; %from MacCormack2
      U1 = alpha .* A(2,:) ./ h; %A(2,:) ./ h; %�P�ʋ���������̉^���ʁiU*h�j
      U2 = A(2,:) ./ h ./ params.s ;
      
      Cr_max = 0.5;
      U_max = abs(max(U2-U1)); %abs(max(U2-U1));
      h_max = max(h);
      dt = Cr_max .* dx ./ (U_max + sqrt(g .* h_max)); %���ԃO���b�h�Ԋu
      lm = dt ./ dx; % ��t/��x

end