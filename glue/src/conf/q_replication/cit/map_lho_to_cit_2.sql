-- This file is part of the Grid LSC User Environment (GLUE)
-- 
-- GLUE is free software: you can redistribute it and/or modify it under the
-- terms of the GNU General Public License as published by the Free Software
-- Foundation, either version 3 of the License, or (at your option) any later
-- version.
-- 
-- This program is distributed in the hope that it will be useful, but WITHOUT
-- ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
-- FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
-- details.
-- 
-- You should have received a copy of the GNU General Public License along
-- with this program.  If not, see <http://www.gnu.org/licenses/>.

--Beginning of script 2--   DatabaseDB2LUOW (SEG_CIT) [WARNING***Please do not alter this line]--
-- CONNECT TO SEG_CIT USER XXXX using XXXX;
INSERT INTO ASN.IBMQREP_RECVQUEUES
 (repqmapname, recvq, sendq, adminq, capture_alias, capture_schema,
 num_apply_agents, memory_limit, state, description, capture_server)
 VALUES
 ('SEG_LHO_ASN_TO_SEG_CIT_ASN', 'ASN.QM1_TO_QM3.DATAQ',
 'ASN.QM1_TO_QM3.DATAQ', 'ASN.QM1.ADMINQ', 'SEG_LHO', 'ASN', 16, 2,
 'A', '', 'SEG_LHO');
-- COMMIT;
