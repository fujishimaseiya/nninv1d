function h_obs = get_obs(para)
% �e�ϑ��n�_�ł̑͐ϗ�
    %num is last examine location point and last location ex)80
    prefix2 = para.prefix2;
%     init_x = load([prefix2 'x_init' '.txt']); %CP�̓ǂݍ���    
    DistResource = para.DistResource;
    noGS = para.noGS;
    
    %% OBS�̃f�[�^�̎擾
%     H = zeros(noGS, size(x_init, 2)); 
%     H = load([params.prefix 'H_test.txt']); %CP�̓ǂݍ���
%     h_obs = zeros(noGS, size(DistResource, 2));

        h_obs(1,:) = load([prefix2 'H1_kiyo.txt']); %CP�̓ǂݍ���
        h_obs(2,:) = load([prefix2 'H2_kiyo.txt']); %CP�̓ǂݍ���
        h_obs(3,:) = load([prefix2 'H3_kiyo.txt']); %CP�̓ǂݍ���
%     if noGS == 1
%         H1 = load([prefix2 'H1.txt']); %CP�̓ǂݍ���
%         H(1,:) = H1(end,:);
%         h_obs = interp1(init_x, H, DistResource); % CP�ւ̕ύX
%     elseif noGS == 2
%         H1 = load([prefix2 'H1.txt']); %CP�̓ǂݍ���
%         H2 = load([prefix2 'H2.txt']); %CP�̓ǂݍ���
%         H(1,:) = H1(end,:);
%         H(2,:) = H2(end,:);
%         h_obs = zeros(noGS, size(DistResource, 2)); % CP�ւ̕ύX
%         for m = 1:noGS
%             h_obs(m,:) = interp1(init_x, H(m,:), DistResource);
%         end
%     elseif noGS == 3
%         H1 = load([prefix2 'H1.txt']); %CP�̓ǂݍ���
%         H2 = load([prefix2 'H2.txt']); %CP�̓ǂݍ���
%         H3 = load([prefix2 'H3.txt']); %CP�̓ǂݍ���
%         H(1,:) = H1(end,:);
%         H(2,:) = H2(end,:);
%         H(3,:) = H3(end,:);
%         h_obs = zeros(noGS, size(DistResource, 2)); % CP�ւ̕ύX
%         for m = 1:noGS
%             h_obs(m,:) = interp1(init_x, H(m,:), DistResource);
%         end
%     end
end
%% memo

%     for m = 1:noGS
%         order = num2str(m);
%         H = load([params.prefix 'H' order '.txt']); %CP�̓ǂݍ���
%     end

% [H, sumH] = count_H(eta_CP, para);

%     %% �����W�̎擾
%     x = load([params.prefix 'x' '.txt']); %�n�`
%     target.x = x;
% 
% %% OBS�̃f�[�^�̎擾
%     for k = 1:noGS
%         order = num2str(k);
%         target.i = load([params.prefix 'eta_' order '.txt' ]); %h_sand���̑͐ϕ���
%         % �e�ϑ��n�_�ł̑͐ϗ�
%         h_obs(order,:) = interp1(target.x, target.i, params.xo, 'pchip', 0); %h_obs(1) = interp1(target.x,target.s,params.xo,'linear', 0);
%     end