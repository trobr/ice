%����б�������ɫͼ���л�ȡ��ͬ��ɫ��ռ���������

clear;
maxImageNum=2000;

%filepath = 'D:\�������ݷ���\����\7�α�����������\�ܼ���ͼ����_sample\';
filepath = 'F:\2����ʦ�����ܼ���2024������\����ʦ�����ܼ���2024��������\D03ͼƬ\RGB\'; 
%������
out = {'Filename', 'water', 'pond', 'ice'};
%����к�
k = 1;

for i = 0:maxImageNum            
    index = ['00000',num2str(i)];
    len_index = length(index);
    imageName = ['RWB_',index(len_index-4:end),'.JPG'];
%   imageName = ['Capture_ (',num2str(i),').tif'];
    filename = [filepath,imageName];
   
    %���ͼ���ļ��Ƿ����
    r=exist(filename);
    if r~=0
        %����ļ����ڣ����ú���getIPPOblique
        [water, pond, ice] = getIPPOblique(filename);
        out(k, :) = {imageName, water, pond, ice};
        k = k+1;
    else
        %����ļ������ڣ���©������
        %out(i+1,:) = {imageName, 0, 0, 0};
    end
    i
end

xlswrite('obliqueAreaFraction.xls', out);