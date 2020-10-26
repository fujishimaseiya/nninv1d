%% �n�`�̌`��_160204
function [x, eta]=make_topo(filename, dx)
    %�f�[�^�̓ǂݍ���
    topodata = load(filename);%filename = topofile
    xo = topodata(:,1);
    yo = topodata(:,2);
    
    %�f�[�^�⊮�ɂ��x���W��y���W�����
    x = xo(1):dx:xo(end);
    eta = interp1(xo,yo,x); % 1���f�[�^���} (�e�[�u�� ���b�N�A�b�v)
    
end