\NeedsTeXFormat{LaTeX2e}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[wcp]{jmlr}

%\jmlrvolume{vol}
%\jmlryear{2012}

\ifanonsubmission
 \newcommand{\coltauthor}[1]{}
 \author{author names withheld}
% \editor{Proposal for term project}
 \jmlrworkshop{Computational Intelligence (MAI) - 2021-22}
 \renewcommand{\acks}[1]{}
\else
% \newcommand{\coltauthor}[1]{\author{#1}}
 \iffinalsubmission
 \fi
\fi



