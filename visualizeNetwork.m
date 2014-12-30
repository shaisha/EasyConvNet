function visualizeNetwork(net,fName)
% visualize network using latex tikz
% Input:
%  net definition
%  fName - name of a latex output file
% The function creates the file fName, in which there's a latex code that
% generates a visualization of the network net

lenO = 0;
for i=1:length(net)
    lenO = max(lenO,max(net{i}.outInd)); 
end

paperheight = lenO*3;

fid = fopen(fName,'wt');

prelatex = {...
'\documentclass[8pt]{article}' , ...
sprintf('\\usepackage[paperwidth=6in, paperheight=%dcm]{geometry}',paperheight) , ...
'\usepackage{tikz}' , ...
'\usetikzlibrary{positioning}' , ...
'\begin{document}'  ,...
' ' , ...
'\begin{tikzpicture}' , ...
'  [nodestyle/.style={rectangle,draw=blue!50,fill=blue!20,thick,' ,...
'                 inner sep=2pt,minimum width=1cm},' , ...
'   ostyle/.style={rectangle,draw=black!50,fill=black!20,thick,' ,...
'                      inner sep=2pt,minimum width=1cm}]' };

fprintf(fid,'%s\n',prelatex{:});


fprintf(fid,'\\node[ostyle] (O1) at (0,0) {1};\n');
for i=2:lenO,
    fprintf(fid,'\\node[ostyle] (O%d) [above=of O%d] {%d};\n',i,i-1,i);
end
fprintf(fid,'\\node[nodestyle] (L1) at (10,0) {%s}',net{1}.type);
for j=1:length(net{1}.outInd)
    fprintf(fid,'\n  edge[->,very thick,blue] (O%d)',net{1}.outInd(j));
end
fprintf(fid,';\n');
for i=2:length(net), 
    fprintf(fid,'\\node[nodestyle] (L%d) [above=of L%d] {%s}',i,i-1,net{i}.type);
    if ~strcmp(net{i}.type,'input')
        for j=1:length(net{i}.inInd)
            fprintf(fid,'\n  edge[<-,very thick,red] (O%d)',net{i}.inInd(j));
        end
    end
    for j=1:length(net{i}.outInd)
        fprintf(fid,'\n  edge[->,very thick,blue] (O%d)',net{i}.outInd(j));
    end
    fprintf(fid,';\n');
end


% closure 
fprintf(fid,'\\end{tikzpicture}\n\\end{document}\n');


fclose(fid);

I = find(fName == '/',1,'last'); coreName = fName(1:end-4); if ~isempty(I), coreName=coreName((I+1):end); end
fprintf(1,'Done. Now run:\n\t unix(''/usr/texbin/pdflatex -output-directory=/tmp %s''); unix(''open /tmp/%s.pdf'');\n',fName,coreName); 

end