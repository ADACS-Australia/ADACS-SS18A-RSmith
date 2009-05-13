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
create table segment_summary
(
segment_sum_id  CHAR(13) FOR BIT DATA NOT NULL,
creator_db      INTEGER NOT NULL WITH DEFAULT 1,
start_time      INTEGER NOT NULL,
end_time        INTEGER NOT NULL,
comment         VARCHAR(255),
segment_def_id  CHAR(13) FOR BIT DATA NOT NULL,
segment_def_cdb INTEGER NOT NULL WITH DEFAULT 1,
process_id      CHAR(13) FOR BIT DATA NOT NULL,

CONSTRAINT seg_sum_pk
      PRIMARY KEY (segment_sum_id, creator_db),
    
      CONSTRAINT seg_sum_fk_pid
      FOREIGN KEY (process_id, creator_db)
          REFERENCES process(process_id, creator_db)
          ON DELETE CASCADE,
          
      CONSTRAINT seg_sum_fk_defid
      FOREIGN KEY (segment_def_id,segment_def_cdb)
          REFERENCES segment_definer(segment_def_id,creator_db)
)

-- The following line is needed for this table to be replicated to other sites
DATA CAPTURE CHANGES
;

CREATE INDEX segsumm_ind_time ON segment_summary(start_time,end_time)
;
CREATE INDEX segsumm_ind_stime ON segment_summary(start_time)
;
CREATE INDEX segsumm_ind_etime ON segment_summary(end_time)
;
CREATE INDEX segsumm_dind ON segment_summary(segment_def_cdb,segment_def_id)
;

