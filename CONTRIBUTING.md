# Contribuindo

Obrigado pelo interesse em melhorar este projeto. Contribuições podem corrigir
exemplos, ampliar a documentação ou adicionar módulos educacionais.

## Antes de começar

1. Procure uma issue existente relacionada à mudança.
2. Para alterações maiores, abra uma issue descrevendo objetivo, abordagem e
   impacto antes de implementar.
3. Não inclua datasets, modelos grandes, credenciais ou conteúdo sem licença
   compatível.

## Fluxo de contribuição

1. Faça um fork do repositório.
2. Crie uma branch a partir de `main`.
3. Mantenha cada pull request focado em um único objetivo.
4. Execute `python -m compileall -q .` antes de enviar.
5. Documente dependências, comandos e resultados reproduzíveis.
6. Abra o pull request usando o template do projeto.

## Padrão para novos módulos

- Explique o conceito e o objetivo educacional no README.
- Use nomes claros para variáveis e funções.
- Fixe seeds aleatórias quando o exemplo comparar métricas.
- Informe origem e licença dos dados utilizados.
- Evite downloads ou treinamentos pesados durante importação do módulo.
- Inclua instruções que funcionem sem GPU sempre que possível.

Ao participar, você concorda em seguir o [Código de Conduta](CODE_OF_CONDUCT.md).
