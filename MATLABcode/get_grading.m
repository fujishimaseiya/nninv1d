function get_grading(prefix, fname, num)
    %% ���ԂƏ����n�`��ǂݍ��ނƃp�����[�^�̐ݒ�
    % eta_CP(k,l,m); eta_CP(time, x_position, noGS)
    
    para.prefix = prefix;
    para.fname = fname;
    para.time = load([prefix 'time.txt']);
    para.init_x = load([prefix 'init_x.txt']);
    para.eta_init= load([prefix 'eta_init.txt']); %init_eta = load([prefix 'init_eta.txt']);
    para.etai_init= load([prefix 'etai_init.txt']);
    para.noGS = num;
    str_noGS = num2str(para.noGS);
    Di = load(['input\' ['InitD' str_noGS] '.txt']); %�͐ϕ��̗��a(m) [1000 * 10^-6; 250 * 10^-6; 62.5 * 10^-6];%3���a
    para.Di = Di .* 10^-6; %���a�̎����𑵂���
%     Di2 = load([prefix 'Di.txt']);
    para.H_unit = 0.005; %0.02; %(m)
    para.divi = 10;

    for m = 1:para.noGS
        order = num2str(m);
        eta_CP(:,:,m) = load([prefix ['eta' order] '.txt' ]); %�͐ϕ��̌����@data(k,l,m) H of control points
    end
    
    %% �͐ϕ��̌����v�Z
    [H] = count_H(eta_CP, para);
    
   %% ���
    % eta_CP(k,l,m); eta_CP(time, x_position, noGS) 
    % time = zeros(size(para.time)+1);

    H_next = zeros(size(H,1)+1,size(H,2),size(H,3)); %1�s�ڂ�0(�͐ρE�N�H�Ȃ�)��������
    H_next(2:end,:,:) = H;
    time = zeros(size(para.time,1)+1,1);%1�s�ڂ�0�b��������
    time(2:end) = para.time; %0�b�̌��time��������
    sumH = sum(H_next,3); %�͐ϕ��̌����̑��a���v�Z
    
    for l = 2: max(find(sumH(end,:))) %upbound��艺�����̎��̃O���b�h����head�̓��B�����܂ł̋�ԃO���b�h����_  %size(sumH,2)
%     while(l <= size(sumH,2) || params.s > init_x(end))
        
        if sumH(end,l) == 0
            continue %���[�v�̈ȉ��̔����֐����n��
        else 
            sumH_temp =  sumH(:,l);
            blockTime = get_blockTime(sumH_temp, time, para);
            for m = 1:para.noGS
                Hi(:,l,m) = interp1(time, H_next(:,l,m), blockTime); %���a�K���Ƃ̑͐ϕ��̌������
            end
            sumH_conp(:,l) = interp1(time, sumH_temp, blockTime); %���a�K���Ƃ̑͐ϕ��̌������
        end
    end
    
    for l = 2: size(Hi,2)
        for m = 1:para.noGS
            for k = 1: para.divi + 1
                GSD(k,l,m) = para.Di(m) * Hi(k,l,m) / sumH_conp(k,l); %���ϗ��a
            end
        end
    end
    
    GSD(isnan(GSD))= 0;
    L = GSD < 0;
    GSD(L) = 0;
    
    sumGSD = sum(GSD,3);
    %% �����o��
    dlmwrite([prefix 'sumGSD' '.txt'], sumGSD); %���ϗ��a
    dlmwrite([prefix 'sumHi' '.txt'], sumH_conp); %�S�̂̑͐ϕ��̌���
end
%% memo
% GSD(:,:,k) = Di(k,1) * fracH(:,:,k);
