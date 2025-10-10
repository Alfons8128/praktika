\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{prepareprotokol}
\LoadClass[10pt,a4paper]{article}

%Balíčky pro celý dokument
\usepackage[utf8]{inputenc}
\usepackage[czech]{babel}
\usepackage[T1]{fontenc}
\usepackage{graphicx}\usepackage{lmodern}
\usepackage[top = 2cm, bottom = 2cm, left = 2cm, right = 2cm]{geometry}

%Balíčky pro titulní stránku
\usepackage{titlesec}
\usepackage{setspace} %Rámeček nahoře
\usepackage{framed} %Rámeček nahoře
\usepackage{array} %Tabulka dole

\newcommand{\praktikum}[1]{\newcommand{\ppraktikum}{#1}}
\newcommand{\autor}[1]{\newcommand{\aautor}{#1}} 
\newcommand{\datum}[1]{\newcommand{\ddatum}{#1}}
\newcommand{\cislo}[1]{\newcommand{\ccislo}{#1}}
\newcommand{\nazev}[1]{\newcommand{\nnazev}{#1}}

\renewcommand*{\maketitle}{%
\begin{titlepage}

\thispagestyle{empty}
\newgeometry{top = 2.5cm, bottom = 0cm, left = 2.5cm, right = 3cm}

{%V tomto je uzavřena celá titulka
%Tloušťka rámečku
\setlength{\fboxrule}{1.5pt}

\noindent
\framebox{
\begin{minipage}{\textwidth}
\setlength{\parindent}{17.62482 pt}
\phantom{d}

\begin{minipage}{0.6\textwidth}
{
\Large Kabinet výuky obecné fyziky, UK MFF\\
}
\vspace*{0.2cm}

{
\bfseries
\huge Fyzikální praktikum \ppraktikum%ČÍSLO
}
\end{minipage}
\begin{minipage}{0.4\textwidth}
\begin{center}
\includegraphics[width=4.5cm]{ZFP.pdf}
\end{center}
\end{minipage}\\\\

%\vspace*{0.5cm}

{
\setstretch{1.5}
\Large
\noindent
Úloha č. \ccislo%Číslo

\noindent
Název úlohy: \nnazev%Název

\noindent
Jméno: \aautor%Jméno

\noindent
Datum měření: \ddatum%Datum

\phantom{d}
}
\end{minipage}
}
%Konec horního rámečku

{
\phantom{d}

\Large
Připomínky opravujícího:\\
\vspace*{6.75cm}
}

\newcommand{\linka}{\noalign{\hrule height 1pt}}
\newcommand{\linkadva}{\noalign{\hrule height 1.5pt}}
\setlength\extrarowheight{9.5pt}
\Large
\noindent
\begin{tabular}{!{\vrule width 1.5pt} l !{\vrule width 1pt} c !{\vrule width 1pt} c !{\vrule width 1.5pt}}
\linkadva
   & Možný počet bodů & Udělený počet bodů \\\linkadva
  Teoretická část a Literatura & 0--1 &  \\\linka
  Výsledky a zpracování měření & 0--7 &  \\\linka
  Diskuse výsledků & 0--3 &  \\\linka
  Závěr & 0--1 &  \\\linkadva
  \hspace*{\fill} \textbf{Celkem} \hspace*{\fill}& max. 12 &  \\
\linkadva
\end{tabular}
\phantom{d}

Posuzoval: \hspace*{\fill}dne:~~~~~~~~~~~~~~~~~

}%Konec uzavření titulky
\end{titlepage}
\newpage
\newgeometry{top = 2cm, bottom = 2cm, left = 2cm, right = 2cm}
\setcounter{page}{1}
}

%
%Šablonu z webových stránek https://physics.mff.cuni.cz/vyuka/zfp/titulky upravil Jiří Zelenka pro psaní protokulů na předmět ZFP na MMF UK (2021).
%Verze: 2.0
%Licence: CC BY-NC-SA