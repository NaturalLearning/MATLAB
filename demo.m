% This code is licensed under the GNU General Public License v3.0
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%
% *********************************************
%
% Implemenation of "Natural Learning"
%
% website: www.natural-learning.cc
% Author: Hadi Fanaee-T
% Associate Professor of Machine Learning
% School of Information Technology
% Halmstad University, Sweden
% Email: hadi.fanaee@hh.se
%
%
% Please cite the following paper if you use the code
%
% *********************************************
% Hadi Fanaee-T, "Natural Learning", arXiv:2404.05903
% https://arxiv.org/abs/2404.05903
%
% *********************************************
% BibTeX
% *********************************************
%
% @article{fanaee2024natural,
%   title={Natural Learning},
%   author={Fanaee-T, Hadi},
%   journal={arXiv preprint arXiv:2404.05903},
%   year={2024}
%}
%



clear
X_train=readmatrix('X_train.csv');
y_train=readmatrix('y_train.csv');
X_test=readmatrix('X_test.csv');
y_test=readmatrix('y_test.csv');
Mdl=fitNL(X_train,y_train);
y_pred=Mdl.My(knnsearch(Mdl.MX,X_test(:,Mdl.PrototypeFeatureIDs),'K',1));
err_test=sum(y_pred~=y_test)/numel(y_test);
disp(sprintf('Test Error = %f',err_test));

