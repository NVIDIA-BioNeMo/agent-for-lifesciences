# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add the server-side conversation record and findings ledger

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-03

Moves ownership of the conversation from the browser to the backend. The
frontend previously sent back only role and content, so tool calls, tool
results, and reasoning were discarded between user turns; conversations,
turns, and messages give each of those a column.

findings is the memory substrate: one row per tool execution, with the
numeric results the domain produces (affinity, pLDDT, ipTM, LogP, TPSA)
extracted into a scalars column so they can be re-rendered deterministically
instead of summarized.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0005'
down_revision: Union[str, None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', sa.String(64), nullable=False),
        sa.Column('model_id', sa.String(64), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_conversations_session_id', 'conversations', ['session_id']
    )

    op.create_table(
        'turns',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['conversation_id'], ['conversations.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_turns_conversation_id', 'turns', ['conversation_id'])

    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('turn_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False, server_default=''),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('tool_calls', postgresql.JSONB(), nullable=True),
        sa.Column('tool_call_id', sa.String(64), nullable=True),
        sa.Column('name', sa.String(128), nullable=True),
        sa.Column('extra', postgresql.JSONB(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['turn_id'], ['turns.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_messages_turn_id', 'messages', ['turn_id'])

    op.create_table(
        'findings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', sa.String(64), nullable=False),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('turn_seq', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('iteration', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tool_name', sa.String(128), nullable=False),
        sa.Column('arguments', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('input_refs', postgresql.JSONB(), nullable=False, server_default='[]'),
        sa.Column('output_refs', postgresql.JSONB(), nullable=False, server_default='[]'),
        sa.Column('scalars', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('summary', sa.Text(), nullable=False, server_default=''),
        sa.Column('success', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['conversation_id'], ['conversations.id'], ondelete='SET NULL'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_findings_session_id', 'findings', ['session_id'])
    op.create_index('ix_findings_conversation_id', 'findings', ['conversation_id'])
    op.create_index('ix_findings_tool_name', 'findings', ['tool_name'])


def downgrade() -> None:
    op.drop_index('ix_findings_tool_name', table_name='findings')
    op.drop_index('ix_findings_conversation_id', table_name='findings')
    op.drop_index('ix_findings_session_id', table_name='findings')
    op.drop_table('findings')

    op.drop_index('ix_messages_turn_id', table_name='messages')
    op.drop_table('messages')

    op.drop_index('ix_turns_conversation_id', table_name='turns')
    op.drop_table('turns')

    op.drop_index('ix_conversations_session_id', table_name='conversations')
    op.drop_table('conversations')
