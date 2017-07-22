
for i=349:382
filename=strcat('OAS1_',num2str(i,'%04d'),'_MR1');
analyzefilename=strcat(filename,'/PROCESSED/MPRAGE/T88_111/',filename,'_mpr_n4_anon_111_t88_masked_gfc');
analyzefilename1=strcat(filename,'/PROCESSED/MPRAGE/T88_111/',filename,'_mpr_n3_anon_111_t88_masked_gfc');
    if exist(strcat(analyzefilename,'.hdr'),'file')
        X = analyze75read(analyzefilename);
    
    elseif exist(strcat(analyzefilename1,'.hdr'),'file')
        X = analyze75read(analyzefilename1);
    else
        i
        continue;
    end  
    imagefoldername=strcat('2Dimages/',filename);
    mkdir(imagefoldername);
    val=zeros(size(X,3),1);
    parfor l=1:size(X,3)
        val(l)=entropy(mat2gray(X(:,:,l))/max(max(mat2gray(X(:,:,l)))));
  
    end

[~,Xind]=sort(val,'descend');
    for j=1:32
        imwrite(mat2gray(X(:,:,Xind(j))),strcat(imagefoldername,'/',num2str(j,'%02d'),'.jpg'));
    end
end

