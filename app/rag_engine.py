import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
import openai
from app.config import settings
from app.vector_store import vector_store_manager
from app.database import get_db, SearchQuery, RAGSession, CacheManager
from sqlalchemy.orm import Session


class RAGEngine:
    def __init__(self):
        # Initialize OpenAI client
        openai.api_key = settings.openai_api_key
        
        # Initialize language model
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=settings.openai_api_key
        )
        
        # Cache manager for responses
        self.cache_manager = CacheManager()
        
        # Default collection name
        self.default_collection = "documents"
        
        # RAG prompts
        self.qa_prompt_template = """
You are an AI assistant that answers questions based on the provided context from documents.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always cite which document or source your answer comes from when possible.

Context:
{context}

Question: {question}

Answer: """

                 self.conversation_prompt_template = """
You are an AI assistant having a conversation with a user about documents in their knowledge base.
Use the following context from relevant documents to inform your responses.
Maintain context from the conversation history and provide helpful, accurate answers.

Previous conversation:
{chat_history}

Current context from documents:
{context}

User: {question}
Assistant: """

        # Initialize prompts
        self.qa_prompt = PromptTemplate(
            template=self.qa_prompt_template,
            input_variables=["context", "question"]
        )
    
    async def answer_question(
        self,
        question: str,
        collection_name: str = None,
        search_type: str = "hybrid",
        k: int = None,
        session_id: str = None,
        filter_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Answer a question using RAG"""
        start_time = time.time()
        collection_name = collection_name or self.default_collection
        k = k or settings.max_retrieval_docs
        
        try:
            # Check cache first
            cache_key = f"rag:{hash(question)}:{collection_name}:{search_type}:{k}"
            cached_result = self.cache_manager.get_cache(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Retrieve relevant documents
            if search_type == "hybrid":
                search_results = await vector_store_manager.hybrid_search(
                    collection_name, question, k, filter_metadata=filter_metadata
                )
            else:
                search_results = await vector_store_manager.similarity_search(
                    collection_name, question, k, filter_metadata=filter_metadata
                )
            
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "response_time": time.time() - start_time
                }
            
            # Prepare context from search results
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results):
                context_parts.append(f"[Source {i+1}]: {result['content']}")
                sources.append({
                    "id": result['id'],
                    "content_preview": result['content'][:200] + "...",
                    "similarity_score": result['similarity_score'],
                    "metadata": result['metadata']
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using LLM
            formatted_prompt = self.qa_prompt.format(context=context, question=question)
            response = await self.llm.agenerate([[HumanMessage(content=formatted_prompt)]])
            answer = response.generations[0][0].text.strip()
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(r['similarity_score'] for r in search_results) / len(search_results)
            confidence = min(avg_similarity * 1.2, 1.0)  # Boost confidence slightly
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "search_results_count": len(search_results),
                "response_time": time.time() - start_time,
                "search_type": search_type
            }
            
            # Cache the result
            self.cache_manager.set_cache(
                cache_key,
                json.dumps(result, default=str),
                expire=1800  # 30 minutes
            )
            
            # Log the query
            await self._log_search_query(question, len(search_results), time.time() - start_time)
            
            # Update session if provided
            if session_id:
                await self._update_rag_session(session_id, question, answer, [r['metadata'].get('document_id') for r in search_results])
            
            return result
            
        except Exception as e:
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def conversational_qa(
        self,
        question: str,
        session_id: str,
        collection_name: str = None,
        k: int = None
    ) -> Dict[str, Any]:
        """Have a conversational Q&A with memory"""
        start_time = time.time()
        collection_name = collection_name or self.default_collection
        k = k or settings.max_retrieval_docs
        
        try:
            # Get or create session
            session = await self._get_or_create_session(session_id)
            
            # Get conversation history
            chat_history = json.loads(session.conversation_history) if session.conversation_history else []
            
            # Retrieve relevant documents
            search_results = await vector_store_manager.hybrid_search(
                collection_name, question, k
            )
            
            # Prepare context
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results):
                context_parts.append(f"[Source {i+1}]: {result['content']}")
                sources.append({
                    "id": result['id'],
                    "content_preview": result['content'][:200] + "...",
                    "similarity_score": result['similarity_score'],
                    "metadata": result['metadata']
                })
            
            context = "\n\n".join(context_parts)
            
            # Format chat history for prompt
            history_text = ""
            for entry in chat_history[-5:]:  # Last 5 exchanges
                history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n\n"
            
            # Generate conversational response
            formatted_prompt = self.conversation_prompt_template.format(
                chat_history=history_text,
                context=context,
                question=question
            )
            
            response = await self.llm.agenerate([[HumanMessage(content=formatted_prompt)]])
            answer = response.generations[0][0].text.strip()
            
            # Update conversation history
            chat_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "sources": [s['id'] for s in sources]
            })
            
            # Update session
            await self._update_session_history(session_id, chat_history, [r['metadata'].get('document_id') for r in search_results])
            
            result = {
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "conversation_length": len(chat_history),
                "response_time": time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            return {
                "answer": f"An error occurred during the conversation: {str(e)}",
                "sources": [],
                "session_id": session_id,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def summarize_document(
        self,
        document_id: int,
        collection_name: str = None,
        summary_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Generate a summary of a specific document"""
        collection_name = collection_name or self.default_collection
        
        try:
            # Get all chunks for the document
            collection = vector_store_manager.chroma_client.get_collection(collection_name)
            results = collection.get(
                where={"document_id": document_id},
                include=['documents', 'metadatas']
            )
            
            if not results['documents']:
                return {"summary": "Document not found.", "error": "Document not found"}
            
            # Combine all chunks
            full_content = "\n\n".join(results['documents'])
            
            # Create summary prompt based on type
            if summary_type == "brief":
                prompt = f"Provide a brief summary (2-3 sentences) of the following document:\n\n{full_content[:4000]}"
            elif summary_type == "detailed":
                prompt = f"Provide a detailed summary with key points and main themes of the following document:\n\n{full_content[:4000]}"
            else:  # comprehensive
                prompt = f"Provide a comprehensive summary including main topics, key insights, and important details from the following document:\n\n{full_content[:4000]}"
            
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            summary = response.generations[0][0].text.strip()
            
            # Extract metadata from first chunk
            metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            return {
                "summary": summary,
                "document_id": document_id,
                "summary_type": summary_type,
                "chunk_count": len(results['documents']),
                "content_length": len(full_content),
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "error": str(e)
            }
    
    async def generate_questions(
        self,
        document_id: int,
        collection_name: str = None,
        num_questions: int = 5
    ) -> Dict[str, Any]:
        """Generate potential questions that could be answered by a document"""
        collection_name = collection_name or self.default_collection
        
        try:
            # Get document content
            collection = vector_store_manager.chroma_client.get_collection(collection_name)
            results = collection.get(
                where={"document_id": document_id},
                include=['documents', 'metadatas'],
                limit=3  # Just get first few chunks for question generation
            )
            
            if not results['documents']:
                return {"questions": [], "error": "Document not found"}
            
            content = "\n\n".join(results['documents'])
            
            prompt = f"""
Based on the following document content, generate {num_questions} meaningful questions that could be answered using this document. 
Make the questions specific and relevant to the content.

Document content:
{content[:3000]}

Generate {num_questions} questions:
"""
            
            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            questions_text = response.generations[0][0].text.strip()
            
            # Parse questions (simple parsing - in production you might want more sophisticated parsing)
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and any(q.strip().endswith(p) for p in ['?', '.'])]
            
            return {
                "questions": questions[:num_questions],
                "document_id": document_id,
                "generated_from_chunks": len(results['documents'])
            }
            
        except Exception as e:
            return {
                "questions": [],
                "error": str(e)
            }
    
    async def _log_search_query(self, query: str, results_count: int, response_time: float):
        """Log search query for analytics"""
        try:
            db = next(get_db())
            search_query = SearchQuery(
                query_text=query,
                query_hash=str(hash(query)),
                results_count=results_count,
                response_time=response_time
            )
            db.add(search_query)
            db.commit()
        except Exception:
            pass  # Don't fail the main operation if logging fails
    
    async def _get_or_create_session(self, session_id: str) -> RAGSession:
        """Get existing session or create new one"""
        db = next(get_db())
        session = db.query(RAGSession).filter(RAGSession.session_id == session_id).first()
        
        if not session:
            session = RAGSession(
                session_id=session_id,
                conversation_history="[]",
                context_documents="[]"
            )
            db.add(session)
            db.commit()
            db.refresh(session)
        
        return session
    
    async def _update_rag_session(self, session_id: str, question: str, answer: str, document_ids: List[int]):
        """Update RAG session with new interaction"""
        try:
            db = next(get_db())
            session = db.query(RAGSession).filter(RAGSession.session_id == session_id).first()
            
            if session:
                session.last_interaction = datetime.now()
                session.total_queries += 1
                
                # Update context documents
                context_docs = json.loads(session.context_documents) if session.context_documents else []
                for doc_id in document_ids:
                    if doc_id and doc_id not in context_docs:
                        context_docs.append(doc_id)
                session.context_documents = json.dumps(context_docs[-10:])  # Keep last 10 documents
                
                db.commit()
        except Exception:
            pass
    
    async def _update_session_history(self, session_id: str, chat_history: List[Dict], document_ids: List[int]):
        """Update session conversation history"""
        try:
            db = next(get_db())
            session = db.query(RAGSession).filter(RAGSession.session_id == session_id).first()
            
            if session:
                session.conversation_history = json.dumps(chat_history)
                session.last_interaction = datetime.now()
                session.total_queries += 1
                
                # Update context documents
                context_docs = json.loads(session.context_documents) if session.context_documents else []
                for doc_id in document_ids:
                    if doc_id and doc_id not in context_docs:
                        context_docs.append(doc_id)
                session.context_documents = json.dumps(context_docs[-10:])
                
                db.commit()
        except Exception:
            pass


# Global RAG engine instance
rag_engine = RAGEngine()