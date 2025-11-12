
"""Extrai Pull Requests e comentários de repositórios GitHub."""

import requests
import json
import sys
from typing import Dict, List, Optional
from datetime import datetime
import time


class GitHubPRExtractor:
    """Extrai Pull Requests e comentários da API do GitHub."""

    API_BASE_URL = "https://api.github.com"
    API_VERSION = "2022-11-28"

    def __init__(self, owner: str, repo: str, token: str):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": self.API_VERSION
        }

    def _requisicao(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Faz requisição à API do GitHub."""
        try:
            resposta = requests.get(url, headers=self.headers, params=params)
            resposta.raise_for_status()

            limite_restante = int(resposta.headers.get('X-RateLimit-Remaining', 0))
            if limite_restante < 10:
                momento_reset = int(resposta.headers.get('X-RateLimit-Reset', 0))
                tempo_espera = max(momento_reset - time.time(), 0) + 1
                print(f"Aguardando rate limit: {tempo_espera:.0f}s...")
                time.sleep(tempo_espera)

            return resposta
        except requests.exceptions.RequestException as e:
            print(f"Erro: {e}", file=sys.stderr)
            raise

    def buscar_prs_fechados(self, por_pagina: int = 100) -> List[Dict]:
        """Busca todos os PRs fechados."""
        print(f"Buscando PRs de {self.owner}/{self.repo}...")
        todos_prs = []
        pagina = 1

        while True:
            url = f"{self.API_BASE_URL}/repos/{self.owner}/{self.repo}/pulls"
            parametros = {"state": "closed", "per_page": por_pagina, "page": pagina}

            print(f"Página {pagina}...", end=" ")
            resposta = self._requisicao(url, parametros)
            prs = resposta.json()

            if not prs:
                break

            todos_prs.extend(prs)
            print(f"{len(prs)} PRs")
            pagina += 1

        print(f"Total: {len(todos_prs)} PRs")
        return todos_prs

    def filtrar_prs_mergeados(self, prs: List[Dict]) -> List[Dict]:
        """Filtra apenas PRs mergeados."""
        mergeados = [pr for pr in prs if pr.get("merged_at")]
        print(f"PRs mergeados: {len(mergeados)}")
        return mergeados

    def buscar_comentarios_gerais(self, numero_pr: int) -> List[Dict]:
        """Busca comentários gerais do PR."""
        url = f"{self.API_BASE_URL}/repos/{self.owner}/{self.repo}/issues/{numero_pr}/comments"
        return self._requisicao(url).json()

    def buscar_comentarios_codigo(self, numero_pr: int) -> List[Dict]:
        """Busca comentários de code review."""
        url = f"{self.API_BASE_URL}/repos/{self.owner}/{self.repo}/pulls/{numero_pr}/comments"
        return self._requisicao(url).json()

    def buscar_reviews(self, numero_pr: int) -> List[Dict]:
        """Busca reviews do PR."""
        url = f"{self.API_BASE_URL}/repos/{self.owner}/{self.repo}/pulls/{numero_pr}/reviews"
        return self._requisicao(url).json()

    def extrair_dados_pr(self, pr: Dict) -> Dict:
        """Extrai dados do PR."""
        return {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "created_at": pr.get("created_at"),
            "merged_at": pr.get("merged_at"),
            "author": pr.get("user", {}).get("login"),
            "body": pr.get("body"),
            "html_url": pr.get("html_url"),
            "additions": pr.get("additions"),
            "deletions": pr.get("deletions"),
            "changed_files": pr.get("changed_files")
        }

    def extrair_dados_comentario(self, comentario: Dict, tipo_comentario: str) -> Dict:
        """Extrai dados do comentário."""
        dados = {
            "type": tipo_comentario,
            "author": comentario.get("user", {}).get("login"),
            "body": comentario.get("body"),
            "created_at": comentario.get("created_at")
        }

        if tipo_comentario == "review":
            dados["path"] = comentario.get("path")
            dados["line"] = comentario.get("line")

        return dados

    def extrair_dados_review(self, review: Dict) -> Dict:
        """Extrai dados do review."""
        return {
            "type": "review_body",
            "author": review.get("user", {}).get("login"),
            "body": review.get("body"),
            "state": review.get("state"),
            "submitted_at": review.get("submitted_at")
        }

    def processar_pull_requests(self, incluir_fechados: bool = False, limite: int = 100) -> Dict:
        """Processa PRs e comentários."""
        todos_prs = self.buscar_prs_fechados()
        prs = todos_prs if incluir_fechados else self.filtrar_prs_mergeados(todos_prs)
        prs = prs[:limite]

        resultado = {
            "repository": f"{self.owner}/{self.repo}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_prs": len(prs),
            "pull_requests": []
        }

        print(f"\nProcessando {len(prs)} PRs...")

        for indice, pr in enumerate(prs, 1):
            numero_pr = pr.get("number")
            print(f"[{indice}/{len(prs)}] PR #{numero_pr}")

            dados_pr = self.extrair_dados_pr(pr)

            comentarios_gerais = self.buscar_comentarios_gerais(numero_pr)
            comentarios_codigo = self.buscar_comentarios_codigo(numero_pr)
            reviews = self.buscar_reviews(numero_pr)

            todos_comentarios = []
            todos_comentarios.extend([self.extrair_dados_comentario(c, "issue") for c in comentarios_gerais])
            todos_comentarios.extend([self.extrair_dados_comentario(c, "review") for c in comentarios_codigo])
            todos_comentarios.extend([self.extrair_dados_review(r) for r in reviews if r.get("body")])

            dados_pr["comments"] = todos_comentarios
            dados_pr["total_comments"] = len(todos_comentarios)
            resultado["pull_requests"].append(dados_pr)

            time.sleep(0.1)

        return resultado

    def salvar_json(self, dados: Dict, nome_arquivo: str = "github_prs_data.json"):
        """Salva dados em JSON."""
        import os
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        caminho = os.path.join(diretorio_atual, nome_arquivo)

        with open(caminho, 'w', encoding='utf-8') as arquivo:
            json.dump(dados, arquivo, indent=2, ensure_ascii=False)

        total_comentarios = sum(pr['total_comments'] for pr in dados['pull_requests'])
        print(f"\n✓ Salvo: {nome_arquivo}")
        print(f"  PRs: {len(dados['pull_requests'])} | Comentários: {total_comentarios}\n")


def main():
    """Executa a extração de PRs."""

    # Configure aqui
    TOKEN_GITHUB = "TOKEN GIT PESSOAL AQUI"
    DONO_REPO = "openai"
    NOME_REPO = "evals"

    try:
        extrator = GitHubPRExtractor(DONO_REPO, NOME_REPO, TOKEN_GITHUB)
        dados = extrator.processar_pull_requests(incluir_fechados=False)

        nome_arquivo = f"{DONO_REPO}_{NOME_REPO}_prs.json"
        extrator.salvar_json(dados, nome_arquivo)

        print("✓ Concluído!")

    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()