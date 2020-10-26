%��v�Ȋ֐�_160204
function [A, A_2nd, params, elapsed_time] = TurbSurge_mlsamples(A, A_2nd, params, interval, endtime)
    %% TurbSurge�𗬂�鍬�����̐��l�v�Z���s��
    %interval: ���b���ƂɌ��ʂ��L�^���邩
    %endtime: �v�Z���I�����鎞��
    %A: ����̌����E�Z�x�ic*h�j�E�����iU*h�j
    %params: �v�Z�ɕK�v�ȃp�����[�^
    %prefix: �v�Z���ʂ�ۑ�����t�@�C���ɕt���閼�O�i�t�H���_���j

%     exec('MacCormack.m');%�}�R�[�}�b�N�@�����s����֐�
%     exec('plot_result.m');%���ʂ�\������֐�
%     exec('get_next_eta.m');%���̃X�e�b�v��eta���v�Z����֐�
%     exec('set_params.m');%���̎��ԃX�e�b�v�ɂ�����e��p�����[�^���v�Z����֐�
%     exec('save_result.m');%���ʂ�ۑ�����֐�
    
    prefix = params.prefix;
    
    %�v�Z���[�v
    tic();
    i = 1;
    
    terminate = false; %�v�Z�I����������̂��߂̕ϐ�
    while(terminate == false)%�I�������𖞂����܂Ōv�Z�𑱂���
        temp = 0;%�C���^�[�o������p�̃J�E���^
        c = 0; %�f�o�b�O�p�ϐ�
        while(temp < interval)%interval�b���ƂɌ��ʂ�ۑ�
            
            params = set_params(A, params);%�p�����[�^���X�V
            % 	termination condition���`�F�b�N����
            terminate = check_terminate(A, endtime, params);
            if terminate %�v�Z�I�������ɒB���Ă����烋�[�v�𔲂���
                break;
            end
                
            A_2nd_moving = CDreal2moving(A_2nd, params); %etai and Fi are interpolated from real CD to moving CD 2016/11/16
            [A, Es_i] = MacCormack2(A, A_2nd_moving, params); %�P���ԃX�e�b�v���̌v�Z���s��            
            [CP_Ci, CP_Es_i] = CDmoving2real(A, params, Es_i);%dep of control points are interpolated %interpolation from moving CD to real CD�@2016/11/16
            A_2nd = predictor_corrector(A_2nd, params, CP_Ci, CP_Es_i);%exner��Fi�𓾂� 2016/10/31
            params.t = params.t + params.dt; %�v�Z�@���Ōo�߂������Ԃ��L�^
            temp = temp + params.dt;%�C���^�[�o������p
            c = c + 1;            
        
        end
        
        i = i + 1;
        
        
    end
    elapsed_time = toc();%���ۂɌv�Z�ɂ�����������
    
end

%% memo
%             params = set_params(A, params);%�p�����[�^���X�V
%             A_2nd_moving = CDreal2moving(A_2nd, params); %etai and Fi are interpolated from real CD to moving CD 2016/11/16
%             [A, Es_i] = MacCormack2(A, A_2nd_moving, params); %�P���ԃX�e�b�v���̌v�Z���s��
%             %params.eta = get_next_eta(A, params); %�n�`�ω����v�Z
%             
%             [CP_Ci, CP_Es_i] = CDmoving2real(A, params, Es_i);%dep of control points are interpolated %interpolation from moving CD to real CD�@2016/11/16
%             A_2nd = predictor_corrector(A_2nd, params, CP_Ci, CP_Es_i);%exner��Fi�𓾂� 2016/10/31
% %             CP_A = CDmoving2realver1(A, params);
% %             A_2nd = predictor_corrector2(A_2nd, params, CP_A);
%             params.t = params.t + params.dt; %�v�Z�@���Ōo�߂������Ԃ��L�^
%             temp = temp + params.dt;%�C���^�[�o������p
