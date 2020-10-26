%�v�Z���ʂ��i�[����֐�
function save_result_mlsamples(A, A_2nd, params)
    %% �v�Z���ʂ����o��
%     num = sprintf('%d', i); %string(i);
    prefix = 'output\';%params.prefix;
    noGS = params.noGS;
    grid = params.grid;

    h = A(1,:); %A��1�s�ڂ���h����肾��
    U = A(2,:) ./ h; %A��1,2�s�ڂ���U����肾��
    logical = U < eps;
    U(logical) = eps;%�����[���̒n�_��eps�Œu�������� U(find(U < eps)) = eps;
    Ci_temp = A(3:2 + noGS,:) ./ repmat(h, noGS, 1);%A��3�s�ڂ���Ci����肾��
    etai = A_2nd(1:noGS,:); %A��5�s�ڂ���eta_sand����肾�� eta_sand = A(5,:); 
    Fi = A_2nd(noGS+1:2 * noGS,:); %Fsand = A(7,:);
    time = params.t;
    x = params.x .* params.s .* params.ho; %����Ԃւ̕ϊ�
%     xo = params.xo; %�����O���b�h
    %% �����ԁE��Ԃɖ߂�
    h = h .* params.ho;
    U = U .* params.Uo;
    Ci = Ci_temp .* repmat(params.Ci_init, 1, grid);%Ci = Ci .* params.Co%n���a�Ɋg��
    Ct = sum(Ci,1); %C_total = Cs + Cm;
    
    etat = sum(etai, 1); %eta_total = eta_sand + eta_mud;
    Ft = sum(Fi, 1);

    time = time .* params.ho ./ params.Uo;
    %% �w�b�h�̂Ƃ����0��t��������
    h(end+1) = 0;
    U(end+1) = 0;
    Ci(:,end+1) = zeros();
    Ct(:,end+1) = zeros();
%     eta_head = interp1([0:params.topodx:(size(params.eta,2) - 1) .* params.topodx], params.eta, params.s .* params.ho) ./ params.ho;

    x(end+1) = x(end);
    %% �n�`�̓ǂݍ���
    eta_init_head = interp1([0:params.topodx:(size(params.eta,2) - 1) * params.topodx],params.eta,x);%head�܂ł̏����n�`
%     eta_init = params.init_eta;%�S�̏����n�`
    etai_init = params.etai_init;
    %% ���ʂ��t�@�C���ɏ�������
    Hi = zeros(noGS, size(params.eta,2));
    for m = 1:noGS
        order = num2str(m);
        Hi(m,:) = etai(m,:) - etai_init(m,:);
        dlmwrite([prefix 'C' order '.txt'], Ci(m,:),'-append');%�͐ϕ��Z�x
        dlmwrite([prefix 'H' order '.txt'], Hi(m,:),'-append'); % �͐ϕ��̌���
        dlmwrite([prefix 'eta' order  '.txt'], etai(m,:),'-append');%�͐ϕ��̌���+�����n�` eta_i(k,:) + init_eta
        dlmwrite([prefix 'F' order  '.txt'], Fi(m,:),'-append');%���̗��x���z        
    end

    Htotal = sum(Hi,1);%�g�[�^���̑͐ϕ��̌���
    dlmwrite([prefix 'Ct' '.txt'], Ct,'-append');%�S�̂̑͐ϕ��Z�x
    dlmwrite([prefix 'Ht' '.txt'],  Htotal,'-append'); %�S�������͐ϕ��̌���
    dlmwrite([prefix 'etat' '.txt'], etat,'-append');%�S�̂̑͐ϕ��̌��� etat + init_eta
    dlmwrite([prefix 'Ft' '.txt'], Ft,'-append');%�S�̗̂��x���z
    
    dlmwrite([prefix 'xi' '.txt'], h + eta_init_head,'-append');%���ۂ̍������̗���Ă����
    dlmwrite([prefix 'flow_h' '.txt'], h,'-append');%���ۂ̍������̗���Ă����

    %dlmwrite([prefix 'xi' '.txt'], h,'-append');%���ۂ̍������̗���Ă����
    dlmwrite([prefix 'U' '.txt'], U,'-append');%����
    
    dlmwrite([prefix 'x' '.txt'], x,'-append');
    dlmwrite([prefix 'time' '.txt'], time, '-append');
end