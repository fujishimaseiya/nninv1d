function terminate = check_terminate(A, endtime, params)
%�v�Z�̏I�������𔻒肷�邽�߂̊֐�

    terminate = false;
    eps = 0.001;
    
     %�v�Z�����U���Ă��邩�`�F�b�N
    if max(max(isnan(A))) == 1 || min(min(isreal(A))) == 0
        terminate = true;
    
    %�w�b�h�̗����E�������������Ȃ肷���Ă��Ȃ���
    elseif min(A(1:2,end)) < eps
        terminate = true;
        
    %�w�b�h�̔Z�x���������Ȃ肷���Ă��Ȃ���
    elseif (sum((A(3:2 + params.noGS,end) .* params.Ci_init),1) / params.Co) < eps
        terminate = true;
        
    %���ꂪ�v�Z�h���C���̏I�[�ɓ��B���Ă��Ȃ���
    elseif params.s * params.ho > params.xo(end)
        terminate = true;
    
    %�v�Z�I�������ƂȂ��Ă��Ȃ���
    elseif params.t > endtime
        terminate = true;
    end
    
end
