%% �V�~�����[�V�������s_160204
function [A, A_2nd, params, elapsed_time] = exec_TurbSurge_mltest_new(prefix, interval, endtime, icond)
    %TurbSurge�ł̍������V�~�����[�V�������v�Z�����{����
    %setfile: �ݒ�t�@�C���̖���
    %interval: ���ʂ����������Ԋu�i�b�j
    %endtime: �v�Z�I�������i�b�j
    %prefix = 'test_output_otadai04\';

    delete([prefix '*.txt']);%�܂��o�̓t�H���_����ɂ���
    rng('shuffle');%����������������
    
    for i = 1:1

        %�����p�����[�^�ݒ�
        [A, A_2nd, params] = set_init_mltest_new(icond, prefix);
        
            %�����p�����[�^��ۑ�����

    init_x = params.xo;
    dlmwrite([prefix 'x_init.txt'], init_x);%�n�`��x�O���b�h
    dlmwrite([prefix 'eta_init.txt'], params.eta_init);%�����n�`
    dlmwrite([prefix 'Di.txt'], params.Di);%���a�K
    for m = 1: params.noGS
        dlmwrite([prefix 'etai_init.txt'], A_2nd(m,:),'-append');%���a�K���Ƃ̑͐ϕ��̌���
    end
    dlmwrite([prefix 'icond.txt'], icond); %initial conditions

        
        %�ړ����W�n�Ɏ��Ԃ�ϊ�
        interval_trans = interval ./ params.ho .* params.Uo; %interval_trans = interval ./ params.ho .* params.Uo;
        endtime_trans = endtime ./ params.ho .* params.Uo; %endtime_trans = endtime ./ params.ho .* params.Uo;
        params.endtime_trans = endtime_trans;
        [A, A_2nd, params, elapsed_time] =  TurbSurge_mltest(A, A_2nd, params, interval_trans, endtime_trans);
        
        %save_result_mltest(A, A_2nd, params);%�v�Z���ʂ��i�[

    end
    
end
